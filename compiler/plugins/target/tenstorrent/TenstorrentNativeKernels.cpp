// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "TenstorrentNativeKernels.h"

#include "iree/schema/tt_native_dispatch.h"
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace mlir::iree_compiler::IREE::HAL {
namespace {
std::string bits(float value) {
  uint32_t raw;
  std::memcpy(&raw, &value, sizeof(raw));
  return "from_bits(" + std::to_string(raw) + "u)";
}

std::string commonSource(const tt_iree_ttex_entry_point_desc_t &ep,
                         const tt_iree_ttex_tensor_desc_t *const *override = nullptr,
                         unsigned count = TT_IREE_NATIVE_TENSOR_COUNT) {
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_NATIVE_MAX_TENSOR_COUNT] = {};
  tt_iree_ttex_native_tensors(&ep, t);
  auto tensors = override ? override : t;
  std::ostringstream s;
  s << R"kernel(
// tt-iree native-v16 descriptor-specialized F32 storage ABI.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
union FloatBits { uint32_t bits; float value; };
inline float from_bits(uint32_t bits) { FloatBits v; v.bits = bits; return v.value; }
inline uint32_t to_bits(float value) { FloatBits v; v.value = value; return v.bits; }
inline uint16_t bf16_bits(float value) {
  uint32_t bits = to_bits(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return (bits >> 16) | 0x40u;
  return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}
inline float round_bf16(float value) { return from_bits(uint32_t(bf16_bits(value)) << 16); }
inline uint32_t face_index(uint32_t row, uint32_t col) {
  return (row / 16) * 512 + (col / 16) * 256 + (row % 16) * 16 + col % 16;
}
constexpr auto accessor_0 = TensorAccessorArgs<0>();
)kernel";
  for (unsigned i = 1; i < count; ++i)
    s << "constexpr auto accessor_" << i << " = TensorAccessorArgs<accessor_"
      << i - 1 << ".next_compile_time_args_offset()>();\n";
  for (unsigned i = 0; i < count; ++i) {
    const auto &d = *(tensors[i] ? tensors[i] : tensors[2]);
    bool bf16 = d.element_type == TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
    unsigned bytes = bf16 ? 2 : 4;
    std::string element = "0u";
    for (unsigned j = 0; j < d.rank; ++j) {
      element = "(" + element + " * " + std::to_string(d.storage_shape[j]) +
                "u + " + std::to_string(d.offset[j]) + "u + x" +
                std::to_string(j) + " * " +
                std::to_string(d.stride[j] ? d.stride[j] : 1) + "u)";
    }
    s << "inline uint32_t byte_" << i
      << "(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3) { return "
      << "get_arg_val<uint32_t>(" << i * 2 + 1 << ") + " << element
      << " * " << bytes << "u; }\n";
    s << "inline float read_" << i
      << "(uint32_t scratch, uint32_t x0, uint32_t x1=0, uint32_t x2=0, "
         "uint32_t x3=0) {\n"
      << "const auto buffer = TensorAccessor(accessor_" << i
      << ", get_arg_val<uint32_t>(" << i * 2 << "));\n"
      << "uint32_t byte = byte_" << i << "(x0,x1,x2,x3);\n"
      << "uint32_t ptr = scratch + byte % 64u;\n"
      << "noc_async_read(buffer.get_noc_addr(byte/2048u,byte%2048u),ptr," << bytes << ");\n"
      << "noc_async_read_barrier(); return "
      << (bf16 ? "from_bits(uint32_t(*reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ptr)) << 16)"
               : "from_bits(*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ptr))") << "; }\n";
    s << "inline void write_" << i
      << "(uint32_t scratch, float value, uint32_t x0, uint32_t x1=0, uint32_t "
         "x2=0, uint32_t x3=0) {\n"
      << "const auto buffer = TensorAccessor(accessor_" << i
      << ", get_arg_val<uint32_t>(" << i * 2 << "));\n"
      << "uint32_t byte = byte_" << i << "(x0,x1,x2,x3);\n"
      << "uint32_t ptr = scratch + byte % 16u;\n"
      << "*reinterpret_cast<volatile tt_l1_ptr " << (bf16 ? "uint16_t" : "uint32_t")
      << "*>(ptr) = " << (bf16 ? "bf16_bits(value)" : "to_bits(value)") << ";\n"
      << "noc_async_write(ptr,buffer.get_noc_addr(byte/2048u,byte%2048u)," << bytes << ");\n"
      << "noc_async_write_barrier(); }\n";
  }
  return s.str();
}

std::string coords(const tt_iree_ttex_tensor_desc_t &d, const std::string &m,
                   const std::string &n) {
  if (d.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHW)
    return n + "," + m + "/" + std::to_string(d.shape[2]) + "u," + m + "%" +
           std::to_string(d.shape[2]) + "u";
  if (d.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR)
    return n + "," + m;
  return m + "," + n;
}

bool unitStride(const tt_iree_ttex_tensor_desc_t &d, unsigned axis) {
  return d.stride[axis] <= 1;
}

std::string bulkTransferSource(const tt_iree_ttex_tensor_desc_t *const *t) {
  std::ostringstream s;
  s << R"kernel(
// Keep transfers inside one interleaved page and the existing 128-byte scratch.
// Matching the low six address bits also satisfies the narrower write alignment.
inline uint32_t span_scratch(uint32_t scratch, uint32_t byte) {
  return ((scratch + 63u) & ~63u) + byte % 64u;
}
inline uint32_t span_bytes(uint32_t scratch, uint32_t byte, uint32_t count, uint32_t element_bytes) {
  uint32_t size = count * element_bytes;
  uint32_t capacity = scratch + 128u - span_scratch(scratch, byte);
  if (size > capacity) size = capacity;
  uint32_t page_remaining = 2048u - byte % 2048u;
  if (size > page_remaining) size = page_remaining;
  return size;
}
)kernel";
  for (unsigned slot = 0; slot < 3; ++slot) {
    bool write = slot == 2;
    bool bf16 = t[slot]->element_type == TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
    unsigned bytes = bf16 ? 2 : 4;
    s << "inline void " << (write ? "write" : "read") << "_span_" << slot
      << "(uint32_t scratch, uint32_t tile, uint32_t byte, uint32_t count, "
         "uint32_t tile_stride) {\n"
      << "const auto buffer = TensorAccessor(accessor_" << slot
      << ", get_arg_val<uint32_t>(" << slot * 2 << "));\n"
      << "while(count) {\n"
      << "uint32_t ptr=span_scratch(scratch,byte), "
      << "size=span_bytes(scratch,byte,count," << bytes << "u);\n";
    if (!write)
      s << "noc_async_read(buffer.get_noc_addr(byte/"
           "2048u,byte%2048u),ptr,size);\n"
        << "noc_async_read_barrier();\n";
    s << "for(uint32_t i=0;i<size/" << bytes << "u;++i) {\n";
    if (bf16 && write)
      s << "*reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ptr+i*2u) = "
           "bf16_bits(*reinterpret_cast<volatile tt_l1_ptr float*>(tile+i*tile_stride));\n";
    else if (bf16)
      s << "*reinterpret_cast<volatile tt_l1_ptr float*>(tile+i*tile_stride) = "
           "from_bits(uint32_t(*reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ptr+i*2u)) << 16);\n";
    else
      s << "*reinterpret_cast<volatile tt_l1_ptr uint32_t*>("
        << (write ? "ptr+i*4u" : "tile+i*tile_stride")
        << ") = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>("
        << (write ? "tile+i*tile_stride" : "ptr+i*4u") << ");\n";
    s << "}\n";
    if (write)
      s << "noc_async_write(ptr,buffer.get_noc_addr(byte/"
           "2048u,byte%2048u),size);\n"
        << "noc_async_write_barrier();\n";
    s << "byte+=size; count-=size/" << bytes << "u; tile+=(size/" << bytes << "u)*tile_stride;\n}\n}\n";
  }
  return s.str();
}

void emitNativeInputLoads(std::ostringstream &s,
                          const tt_iree_ttex_entry_point_desc_t &ep,
                          const tt_iree_ttex_tensor_desc_t *const *t,
                          const std::string &lhsCoords,
                          const std::string &rhsCoords) {
  bool convolution = ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION;
  const auto &lhs = *t[0];
  const auto &rhs = *t[1];
  auto convIndices = [&]() {
    if (convolution)
      s << "uint32_t ic=k/" << ep.conv2d.kernel_h * ep.conv2d.kernel_w
        << "u,kh=(k/" << ep.conv2d.kernel_w << "u)%" << ep.conv2d.kernel_h
        << "u,kw=k%" << ep.conv2d.kernel_w << "u;\n";
  };
  s << "for(uint32_t i=0;i<1024;++i) { a[i]=0.0f; b[i]=0.0f; }\n";
  bool spatial = lhs.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHW && lhs.rank == 3;
  bool channel =
      lhs.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR && lhs.rank == 2;
  bool rowMajor =
      lhs.layout == TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR && lhs.rank == 2;
  if ((spatial || channel) && unitStride(lhs, lhs.rank - 1) &&
      (!convolution || ep.conv2d.stride_w == 1)) {
    s << "for(uint32_t col=0;col<valid_k;++col) {\n"
      << "uint32_t k=block*32+col;\n";
    convIndices();
    s << "for(uint32_t row=0;row<valid_m;) {\n"
      << "uint32_t m=m_base+row, chunk=valid_m-row;\n";
    if (spatial) {
      uint32_t width = convolution ? ep.conv2d.output.shape[2] : lhs.shape[2];
      s << "uint32_t row_remaining=" << width << "u-m%" << width << "u;\n"
        << "if(chunk>row_remaining) chunk=row_remaining;\n";
    }
    s << "read_span_0(scratch,lhs.get_write_ptr()+(row*32+col)*4u,byte_0("
      << lhsCoords << (spatial ? ",0" : ",0,0")
      << "),chunk,128u);\nrow+=chunk;\n}\n}\n";
  } else if (rowMajor && !convolution && unitStride(lhs, 1)) {
    s << "for(uint32_t row=0;row<valid_m;++row) {\n"
      << "uint32_t m=m_base+row,k=block*32;\n"
      << "read_span_0(scratch,lhs.get_write_ptr()+row*128u,byte_0(m,k,0,0),"
         "valid_k,4u);\n}\n";
  } else {
    s << "for(uint32_t row=0;row<valid_m;++row) "
         "for(uint32_t col=0;col<valid_k;++col) {\n"
      << "uint32_t m=m_base+row,k=block*32+col;\n";
    convIndices();
    s << "a[row*32+col]=read_0(scratch," << lhsCoords << ");\n}\n";
  }
  bool contiguousRhs = rhs.rank == 2 && !convolution && unitStride(rhs, 1);
  if (convolution && rhs.rank == 4)
    contiguousRhs = unitStride(rhs, 1) && unitStride(rhs, 2) &&
                    unitStride(rhs, 3) && rhs.offset[2] == 0 &&
                    rhs.offset[3] == 0 &&
                    rhs.storage_shape[2] == ep.conv2d.kernel_h &&
                    rhs.storage_shape[3] == ep.conv2d.kernel_w;
  if (contiguousRhs) {
    s << "for(uint32_t col=0;col<valid_n;++col) {\n"
      << "uint32_t n=n_base+col,k=block*32;\n";
    convIndices();
    s << "read_span_1(scratch,rhs.get_write_ptr()+col*4u,byte_1(" << rhsCoords;
    if (!convolution)
      s << ",0,0";
    s << "),valid_k,128u);\n}\n";
  } else {
    s << "for(uint32_t row=0;row<valid_k;++row) "
         "for(uint32_t col=0;col<valid_n;++col) {\n"
      << "uint32_t k=block*32+row,n=n_base+col;\n";
    convIndices();
    s << "b[row*32+col]=read_1(scratch," << rhsCoords << ");\n}\n";
  }
}

void emitNativeOutputStores(std::ostringstream &s,
                            const tt_iree_ttex_tensor_desc_t &d) {
  bool spatial = d.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHW && d.rank == 3;
  bool channel =
      d.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR && d.rank == 2;
  if ((spatial || channel) && unitStride(d, d.rank - 1)) {
    s << "for(uint32_t col=0;col<valid_n;++col) {\n"
      << "uint32_t n=n_base+col;\n"
      << "for(uint32_t row=0;row<valid_m;) {\n"
      << "uint32_t m=m_base+row,chunk=valid_m-row;\n";
    if (spatial)
      s << "uint32_t row_remaining=" << d.shape[2] << "u-m%" << d.shape[2]
        << "u;\nif(chunk>row_remaining) chunk=row_remaining;\n";
    s << "write_span_2(scratch,accumulator.get_write_ptr()+(row*32+col)*4u,"
         "byte_2("
      << coords(d, "m", "n");
    s << (spatial ? ",0" : ",0,0") << "),chunk,128u);\nrow+=chunk;\n}\n}\n";
  } else if (d.layout == TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR && d.rank == 2 &&
             unitStride(d, 1)) {
    s << "for(uint32_t row=0;row<valid_m;++row) {\n"
      << "uint32_t m=m_base+row;\n"
      << "write_span_2(scratch,accumulator.get_write_ptr()+row*128u,"
         "byte_2(m,n_base,0,0),valid_n,4u);\n}\n";
  } else {
    s << "for(uint32_t row=0;row<valid_m;++row) "
         "for(uint32_t col=0;col<valid_n;++col) {\n"
      << "uint32_t m=m_base+row,n=n_base+col;\n"
      << "write_2(scratch,acc[row*32+col]," << coords(d, "m", "n") << ");\n}\n";
  }
}

void embeddedVectors(std::ostringstream &s, size_t count, const float *scales,
                     const float *biases) {
  for (unsigned which = 0; which < 2; ++which) {
    s << "static const uint32_t embedded_" << (which ? "bias" : "scale")
      << "[] = {";
    for (size_t i = 0; i < count; ++i) {
      uint32_t raw;
      float value = which ? biases[i] : scales[i];
      std::memcpy(&raw, &value, sizeof(raw));
      s << raw << "u,";
    }
    s << "};\n";
  }
}

std::string channelEpilogueSetup(const tt_iree_ttex_entry_point_desc_t &ep) {
  std::ostringstream s;
  uint32_t flags = 0;
  float common = 1.0f;
  if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE) {
    flags = ep.matmul_like.epilogue_flags;
    common = ep.matmul_like.epilogue_common_scale;
  } else if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION) {
    flags = ep.conv2d.epilogue_flags;
    common = ep.conv2d.epilogue_common_scale;
  }
  if ((flags & TT_IREE_TTEX_EPILOGUE_AFFINE) &&
      (flags & TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE)) {
    s << "float channel_scale = " << bits(common) << " * read_4(scratch,n);\n"
      << "float channel_bias = read_5(scratch,n) - read_3(scratch,n) * "
         "channel_scale;\n";
  }
  if ((flags & TT_IREE_TTEX_EPILOGUE_RESIDUAL) &&
      (flags & TT_IREE_TTEX_EPILOGUE_RESIDUAL_PER_CHANNEL)) {
    s << "float channel_residual_scale = "
      << bits(ep.matmul_like.residual_common_scale) << " * read_8(scratch,n);\n"
      << "float channel_residual_bias = read_9(scratch,n) - read_7(scratch,n) "
         "* "
         "channel_residual_scale;\n";
  }
  return s.str();
}

std::string epilogue(const tt_iree_ttex_entry_point_desc_t &ep,
                     bool reuse_channel = false) {
  std::ostringstream s;
  uint32_t flags = 0;
  float scale = 1, bias = 0, common = 1;
  if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_LINEAR) {
    return "value += read_5(scratch, m, n);\n";
  } else if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE) {
    flags = ep.matmul_like.epilogue_flags;
    scale = ep.matmul_like.epilogue_scale;
    bias = ep.matmul_like.epilogue_bias;
    common = ep.matmul_like.epilogue_common_scale;
  } else {
    flags = ep.conv2d.epilogue_flags;
    scale = ep.conv2d.epilogue_scale;
    bias = ep.conv2d.epilogue_bias;
    common = ep.conv2d.epilogue_common_scale;
  }
  if (flags & TT_IREE_TTEX_EPILOGUE_AFFINE) {
    if (flags & TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE) {
      if (reuse_channel) {
        s << "value = value * channel_scale + channel_bias;\n";
      } else {
        s << "float scale = " << bits(common) << " * read_4(scratch,n);\n"
          << "value = value * scale + (read_5(scratch,n) - read_3(scratch,n) * "
             "scale);\n";
      }
    } else if (flags & TT_IREE_TTEX_EPILOGUE_PER_CHANNEL) {
      s << "value = value * from_bits(embedded_scale[n]) + "
           "from_bits(embedded_bias[n]);\n";
    } else {
      s << "value = value * " << bits(scale) << " + " << bits(bias) << ";\n";
    }
  }
  if (flags & TT_IREE_TTEX_EPILOGUE_RESIDUAL) {
    const auto &d = ep.matmul_like;
    s << "float residual = read_6(scratch," << coords(d.residual, "m", "n")
      << ");\n";
    if (flags & TT_IREE_TTEX_EPILOGUE_RESIDUAL_PER_CHANNEL) {
      if (reuse_channel) {
        s << "residual = residual * channel_residual_scale + "
             "channel_residual_bias;\n";
      } else {
        s << "float residual_scale = " << bits(d.residual_common_scale)
          << " * read_8(scratch,n);\n"
          << "residual = residual * residual_scale + (read_9(scratch,n) - "
             "read_7(scratch,n) * residual_scale);\n";
      }
    } else {
      s << "residual = residual * " << bits(d.residual_scale) << " + "
        << bits(d.residual_bias) << ";\n";
    }
    s << "value += residual;\n";
  }
  if (flags & TT_IREE_TTEX_EPILOGUE_RELU)
    s << "if (value <= 0.0f) value = 0.0f;\n";
  return s.str();
}

constexpr char kCompute[] = R"kernel(
// tt-iree native-v16: exact signed INT8 digit products, INT32 partials.
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
  uint32_t tiles = get_arg_val<uint32_t>(0);
  uint32_t blocks = get_arg_val<uint32_t>(1);
  CircularBuffer lhs(tt::CBIndex::c_0), rhs(tt::CBIndex::c_1), out(tt::CBIndex::c_16);
  CircularBuffer lhs_mid(tt::CBIndex::c_4), rhs_mid(tt::CBIndex::c_5);
  CircularBuffer lhs_lo(tt::CBIndex::c_6), rhs_lo(tt::CBIndex::c_7);
  CircularBuffer lhs_last(tt::CBIndex::c_10), rhs_last(tt::CBIndex::c_11);
  compute_kernel_hw_startup<SrcOrder::Reverse>(tt::CBIndex::c_0,tt::CBIndex::c_1,tt::CBIndex::c_16);
  matmul_init(tt::CBIndex::c_0,tt::CBIndex::c_1);
  const uint32_t a_ids[] = {0,4,6,10}, b_ids[] = {1,5,7,11};
  for (uint32_t tile = 0; tile < tiles; ++tile) {
    for (uint32_t block = 0; block < blocks; ++block) {
      lhs.wait_front(1); rhs.wait_front(1);
      lhs_mid.wait_front(1); rhs_mid.wait_front(1);
      lhs_lo.wait_front(1); rhs_lo.wait_front(1);
      lhs_last.wait_front(1); rhs_last.wait_front(1);
      for(uint32_t a=0;a<4;++a) for(uint32_t b=0;b<4;++b) {
        tile_regs_acquire();
        matmul_tiles(a_ids[a],b_ids[b],0,0,0);
        tile_regs_commit(); tile_regs_wait();
        out.reserve_back(1); pack_tile(0,tt::CBIndex::c_16); out.push_back(1);
        tile_regs_release();
      }
      lhs.pop_front(1); rhs.pop_front(1);
      lhs_mid.pop_front(1); rhs_mid.pop_front(1);
      lhs_lo.pop_front(1); rhs_lo.pop_front(1);
      lhs_last.pop_front(1); rhs_last.pop_front(1);
    }
  }
}
)kernel";
} // namespace

static TenstorrentNativeSources generateNativeSourcesImpl(
    const tt_iree_ttex_entry_point_desc_t &ep, bool use_tensix,
    const tt_iree_ttex_tensor_desc_t *const *overrides = nullptr,
    const std::string *custom_epilogue = nullptr) {
  TenstorrentNativeSources result;
  std::string common = commonSource(ep, overrides,
      overrides ? TT_IREE_BF16_NATIVE_TENSOR_COUNT : TT_IREE_NATIVE_TENSOR_COUNT);
  if (!use_tensix) {
    common.replace(common.find("native-v16"), 10, "native-v15");
  }
  common =
      "#pragma GCC optimize (\"no-fast-math\", \"fp-contract=off\")\n" + common;
  if (!tt_iree_ttex_is_native_matrix_kind(ep.dispatch_kind)) {
    std::ostringstream s;
    s << common << R"kernel(
void kernel_main() {
  CircularBuffer scratch_cb(tt::CBIndex::c_2);
  scratch_cb.reserve_back(1);
  uint32_t scratch = scratch_cb.get_write_ptr();
  uint32_t start = get_arg_val<uint32_t>(20), count = get_arg_val<uint32_t>(21);
  for (uint32_t i = start; i < start + count; ++i) {
)kernel";
    if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL) {
      const auto &p = ep.max_pool;
      s << "uint32_t c=i/" << p.output.shape[1] * p.output.shape[2]
        << "u, h=(i/" << p.output.shape[2] << "u)%" << p.output.shape[1]
        << "u, w=i%" << p.output.shape[2] << "u;\n"
        << "float value = from_bits(0xff800000u);\n"
        << "for(uint32_t kh=0;kh<" << p.kernel_h << "u;++kh)\n"
        << "for(uint32_t kw=0;kw<" << p.kernel_w << "u;++kw) {\n"
        << "float v=read_0(scratch,c,h*" << p.stride_h << "u+kh*"
        << p.dilation_h << "u,w*" << p.stride_w << "u+kw*" << p.dilation_w
        << "u);\n"
        << "if ((to_bits(value)&0x7fffffffu)>0x7f800000u || "
           "(to_bits(v)&0x7fffffffu)>0x7f800000u) "
           "value=from_bits(0x7fc00000u); else if(v>value) value=v; else "
           "if(v==0.0f && value==0.0f) "
           "value=from_bits(to_bits(v)&to_bits(value));\n}\n"
        << "write_2(scratch,value,c,h,w);\n";
    } else if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM) {
      const auto &r = ep.global_sum;
      s << "float value=0.0f;\nfor(uint32_t h=0;h<" << r.reduce_h
        << "u;++h) for(uint32_t w=0;w<" << r.reduce_w
        << "u;++w) value += read_0(scratch,i,h,w);\n"
        << "write_2(scratch,value,i);\n";
    } else if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV) {
      s << "write_2(scratch,read_0(scratch,i)/read_1(scratch,0),i);\n";
    } else {
      throw std::runtime_error("unsupported native descriptor kind");
    }
    s << "}\n";
    if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM)
      s << "if(get_arg_val<uint32_t>(22)==0) write_1(scratch,"
        << bits(ep.global_sum.divisor_value) << ",0);\n";
    s << "}\n";
    result.reader = s.str();
    return result;
  }

  const tt_iree_ttex_tensor_desc_t *default_t[TT_IREE_NATIVE_MAX_TENSOR_COUNT] = {};
  tt_iree_ttex_native_tensors(&ep, default_t);
  auto t = overrides ? overrides : default_t;
  uint32_t m, n, k;
  std::string lhsCoords, rhsCoords;
  bool convolution = ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION;
  if (convolution) {
    const auto &c = ep.conv2d;
    m = c.output.shape[1] * c.output.shape[2];
    n = c.output.shape[0];
    k = c.input.shape[0] * c.kernel_h * c.kernel_w;
    lhsCoords = "ic,(m/" + std::to_string(c.output.shape[2]) + "u)*" +
                std::to_string(c.stride_h) + "u+kh*" +
                std::to_string(c.dilation_h) + "u,(m%" +
                std::to_string(c.output.shape[2]) + "u)*" +
                std::to_string(c.stride_w) + "u+kw*" +
                std::to_string(c.dilation_w) + "u";
    rhsCoords = "n,ic,kh,kw";
  } else if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_LINEAR) {
    m = ep.linear.m;
    n = ep.linear.n;
    k = ep.linear.k;
    lhsCoords = "m,k";
    rhsCoords = "n,k";
  } else {
    m = ep.matmul_like.m;
    n = ep.matmul_like.n;
    k = ep.matmul_like.k;
    lhsCoords = coords(*t[0], "m", "k");
    rhsCoords = "n,k";
  }
  if (!use_tensix) {
    std::ostringstream s;
    s << common << bulkTransferSource(t);
    if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE &&
        ep.matmul_like.epilogue_channel_count)
      embeddedVectors(s, ep.matmul_like.epilogue_channel_count,
                      ep.matmul_like.epilogue_scales,
                      ep.matmul_like.epilogue_biases);
    if (convolution && ep.conv2d.epilogue_channel_count)
      embeddedVectors(s, ep.conv2d.epilogue_channel_count,
                      ep.conv2d.epilogue_scales, ep.conv2d.epilogue_biases);
    s << R"kernel(
// Device RISC-V tiled FP32 reference math. No host tensor computation.
void kernel_main() {
  CircularBuffer lhs(tt::CBIndex::c_0), rhs(tt::CBIndex::c_1);
  CircularBuffer accumulator(tt::CBIndex::c_16), scratch_cb(tt::CBIndex::c_2);
  lhs.reserve_back(1); rhs.reserve_back(1); accumulator.reserve_back(1); scratch_cb.reserve_back(1);
  uint32_t scratch = scratch_cb.get_write_ptr();
  auto a = reinterpret_cast<volatile tt_l1_ptr float*>(lhs.get_write_ptr());
  auto b = reinterpret_cast<volatile tt_l1_ptr float*>(rhs.get_write_ptr());
  auto acc = reinterpret_cast<volatile tt_l1_ptr float*>(accumulator.get_write_ptr());
  uint32_t m_start=get_arg_val<uint32_t>(20), n_start=get_arg_val<uint32_t>(21);
  uint32_t m_count=get_arg_val<uint32_t>(22), n_count=get_arg_val<uint32_t>(23);
  for(uint32_t tm=0;tm<m_count;++tm) for(uint32_t tn=0;tn<n_count;++tn) {
    uint32_t m_base=(m_start+tm)*32, n_base=(n_start+tn)*32;
)kernel"
      << "uint32_t valid_m=" << m << "u-m_base; if(valid_m>32) valid_m=32;\n"
      << "uint32_t valid_n=" << n << "u-n_base; if(valid_n>32) valid_n=32;\n"
      << "for(uint32_t i=0;i<1024;++i) acc[i]=0.0f;\n"
      << "for(uint32_t block=0;block<" << (k + 31) / 32 << "u;++block) {\n"
      << "uint32_t valid_k=" << k << "u-block*32; if(valid_k>32) valid_k=32;\n";
    emitNativeInputLoads(s, ep, t, lhsCoords, rhsCoords);
    s << "for(uint32_t row=0;row<valid_m;++row) for(uint32_t "
         "col=0;col<valid_n;++col) {\n"
      << "float value=acc[row*32+col];\n"
      << "for(uint32_t inner=0;inner<valid_k;++inner) value += a[row*32+inner] "
         "* b[inner*32+col];\n"
      << "acc[row*32+col]=value;\n}\n}\n";
    std::string channel_setup = custom_epilogue ? "" : channelEpilogueSetup(ep);
    if (!channel_setup.empty()) {
      // Only per-channel invariant expressions move; each output's FP32
      // product/add/subtract order and the per-element residual read remain.
      s << "for(uint32_t col=0;col<valid_n;++col) {\n"
        << "uint32_t n=n_base+col;\n"
        << channel_setup << "for(uint32_t row=0;row<valid_m;++row) {\n"
        << "uint32_t m=m_base+row; float value=acc[row*32+col];\n"
        << epilogue(ep, true) << "acc[row*32+col]=value;\n}\n}\n";
    } else {
      s << "for(uint32_t row=0;row<valid_m;++row) for(uint32_t "
           "col=0;col<valid_n;++col) {\n"
        << "uint32_t m=m_base+row,n=n_base+col; float value=acc[row*32+col];\n"
        << (custom_epilogue ? *custom_epilogue : epilogue(ep))
        << "acc[row*32+col]=value;\n}\n";
    }
    emitNativeOutputStores(s, *t[2]);
    s << "}\n}\n";
    result.reader = s.str();
    return result;
  }
  std::ostringstream reader;
  reader
      << common << R"kernel(
// Four base-128 digits span 28 bits relative to each row/column block scale.
inline float digit_scale(uint32_t exponent) {
  if (!exponent) return 1.0f;
  // Mark unsupported exponent ranges for device F32 exceptional evaluation.
  if (exponent <= 6u || exponent >= 254u) return from_bits(0x7fc00000u);
  return from_bits((exponent - 6u) << 23);
}
inline void write_digits(float value, float scale, uint32_t index,
                         volatile tt_l1_ptr uint8_t* d0,
                         volatile tt_l1_ptr uint8_t* d1,
                         volatile tt_l1_ptr uint8_t* d2,
                         volatile tt_l1_ptr uint8_t* d3) {
  volatile tt_l1_ptr uint8_t* digits[] = {d0,d1,d2,d3};
  uint32_t exponent = (to_bits(scale) >> 23) & 255u;
  float remaining = exponent == 255u ? 0.0f : value * from_bits((254u-exponent) << 23);
  for (uint32_t i=0;i<4;++i) {
    int32_t digit=static_cast<int32_t>(remaining);
    uint32_t magnitude=digit < 0 ? -digit : digit;
    digits[i][index]=static_cast<uint8_t>(magnitude | (digit < 0 ? 128u : 0u));
    remaining=(remaining-static_cast<float>(digit))*128.0f;
  }
}
void kernel_main() {
  CircularBuffer lhs(tt::CBIndex::c_0), rhs(tt::CBIndex::c_1), scratch_cb(tt::CBIndex::c_2);
  CircularBuffer lhs_mid(tt::CBIndex::c_4), rhs_mid(tt::CBIndex::c_5);
  CircularBuffer lhs_lo(tt::CBIndex::c_6), rhs_lo(tt::CBIndex::c_7);
  CircularBuffer lhs_last(tt::CBIndex::c_10), rhs_last(tt::CBIndex::c_11);
  CircularBuffer raw_lhs(tt::CBIndex::c_8), raw_rhs(tt::CBIndex::c_9), metadata(tt::CBIndex::c_12);
  scratch_cb.reserve_back(1); uint32_t scratch=scratch_cb.get_write_ptr();
  raw_lhs.reserve_back(1); raw_rhs.reserve_back(1);
  auto raw_a=reinterpret_cast<volatile tt_l1_ptr float*>(raw_lhs.get_write_ptr());
  auto raw_b=reinterpret_cast<volatile tt_l1_ptr float*>(raw_rhs.get_write_ptr());
  uint32_t m_start=get_arg_val<uint32_t>(20), n_start=get_arg_val<uint32_t>(21);
  uint32_t m_count=get_arg_val<uint32_t>(22), n_count=get_arg_val<uint32_t>(23);
  for(uint32_t tm=0;tm<m_count;++tm) for(uint32_t tn=0;tn<n_count;++tn)
)kernel"
      << "for(uint32_t block=0;block<" << (k + 31) / 32 << "u;++block) {\n"
      << "lhs.reserve_back(1); rhs.reserve_back(1);\n"
      << "lhs_mid.reserve_back(1); rhs_mid.reserve_back(1); "
         "lhs_lo.reserve_back(1); rhs_lo.reserve_back(1);\n"
      << "lhs_last.reserve_back(1); rhs_last.reserve_back(1); "
         "metadata.reserve_back(1);\n"
      << "auto scales=reinterpret_cast<volatile tt_l1_ptr "
         "float*>(metadata.get_write_ptr());\n"
      << "for(uint32_t i=0;i<64;++i) scales[i]=0.0f;\n"
      << "auto a=reinterpret_cast<volatile tt_l1_ptr "
         "uint8_t*>(lhs.get_write_ptr());\n"
      << "auto b=reinterpret_cast<volatile tt_l1_ptr "
         "uint8_t*>(rhs.get_write_ptr());\n"
      << "auto a_mid=reinterpret_cast<volatile tt_l1_ptr "
         "uint8_t*>(lhs_mid.get_write_ptr());\n"
      << "auto b_mid=reinterpret_cast<volatile tt_l1_ptr "
         "uint8_t*>(rhs_mid.get_write_ptr());\n"
      << "auto a_lo=reinterpret_cast<volatile tt_l1_ptr "
         "uint8_t*>(lhs_lo.get_write_ptr());\n"
      << "auto b_lo=reinterpret_cast<volatile tt_l1_ptr "
         "uint8_t*>(rhs_lo.get_write_ptr());\n"
      << "auto a_last=reinterpret_cast<volatile tt_l1_ptr "
         "uint8_t*>(lhs_last.get_write_ptr());\n"
      << "auto b_last=reinterpret_cast<volatile tt_l1_ptr "
         "uint8_t*>(rhs_last.get_write_ptr());\n"
      << "for(uint32_t row=0;row<32;++row) for(uint32_t col=0;col<32;++col) {\n"
      << "uint32_t m=(m_start+tm)*32+row, k=block*32+col; float av=0.0f;\n"
      << "if(m<" << m << "u && k<" << k << "u) {\n";
  if (convolution)
    reader << "uint32_t ic=k/" << ep.conv2d.kernel_h * ep.conv2d.kernel_w
           << "u, kh=(k/" << ep.conv2d.kernel_w << "u)%" << ep.conv2d.kernel_h
           << "u, kw=k%" << ep.conv2d.kernel_w << "u;\n";
  reader << "av=read_0(scratch," << lhsCoords << "); }\n"
         << "uint32_t index=face_index(row,col); raw_a[index]=av;\n"
         << "uint32_t ea=(to_bits(av)>>23)&255u; if(!ea && "
            "(to_bits(av)&0x7fffffu)) ea=255u;\n"
         << "if(ea>to_bits(scales[row])) "
            "scales[row]=from_bits(ea);\n"
         << "k=block*32+row; uint32_t n=(n_start+tn)*32+col; float bv=0.0f;\n"
         << "if(n<" << n << "u && k<" << k << "u) {\n";
  if (convolution)
    reader << "uint32_t ic=k/" << ep.conv2d.kernel_h * ep.conv2d.kernel_w
           << "u, kh=(k/" << ep.conv2d.kernel_w << "u)%" << ep.conv2d.kernel_h
           << "u, kw=k%" << ep.conv2d.kernel_w << "u;\n";
  reader
      << "bv=read_1(scratch," << rhsCoords << "); }\n"
      << "raw_b[index]=bv;\n"
      << "uint32_t eb=(to_bits(bv)>>23)&255u; if(!eb && "
         "(to_bits(bv)&0x7fffffu)) eb=255u;\n"
      << "if(eb>to_bits(scales[32+col])) "
         "scales[32+col]=from_bits(eb);\n"
      << "}\nfor(uint32_t i=0;i<64;++i) "
         "scales[i]=digit_scale(to_bits(scales[i]));\n"
      << "for(uint32_t row=0;row<32;++row) for(uint32_t col=0;col<32;++col) {\n"
      << "uint32_t index=face_index(row,col);\n"
      << "write_digits(raw_a[index],scales[row],index,a,a_mid,a_lo,a_last);\n"
      << "write_digits(raw_b[index],scales[32+col],index,b,b_mid,b_lo,b_last);"
         "\n}\n"
      << "metadata.push_back(1); lhs.push_back(1); rhs.push_back(1); "
         "lhs_mid.push_back(1); "
         "rhs_mid.push_back(1); lhs_lo.push_back(1); "
         "rhs_lo.push_back(1); lhs_last.push_back(1); "
         "rhs_last.push_back(1);\n}\n}\n";
  result.reader = reader.str();

  std::ostringstream writer;
  writer << common;
  if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE &&
      ep.matmul_like.epilogue_channel_count)
    embeddedVectors(writer, ep.matmul_like.epilogue_channel_count,
                    ep.matmul_like.epilogue_scales,
                    ep.matmul_like.epilogue_biases);
  if (convolution && ep.conv2d.epilogue_channel_count)
    embeddedVectors(writer, ep.conv2d.epilogue_channel_count,
                    ep.conv2d.epilogue_scales, ep.conv2d.epilogue_biases);
  writer
      << R"kernel(
void kernel_main() {
  CircularBuffer out(tt::CBIndex::c_16), scratch_cb(tt::CBIndex::c_3);
  CircularBuffer sum_cb(tt::CBIndex::c_17), compensation_cb(tt::CBIndex::c_18);
  CircularBuffer metadata(tt::CBIndex::c_12);
  scratch_cb.reserve_back(1); uint32_t scratch=scratch_cb.get_write_ptr();
  sum_cb.reserve_back(1); compensation_cb.reserve_back(1);
  auto sum=reinterpret_cast<volatile tt_l1_ptr float*>(sum_cb.get_write_ptr());
  auto compensation=reinterpret_cast<volatile tt_l1_ptr float*>(compensation_cb.get_write_ptr());
  uint32_t m_start=get_arg_val<uint32_t>(20), n_start=get_arg_val<uint32_t>(21);
  uint32_t m_count=get_arg_val<uint32_t>(22), n_count=get_arg_val<uint32_t>(23);
  for(uint32_t tm=0;tm<m_count;++tm) for(uint32_t tn=0;tn<n_count;++tn) {
)kernel"
      << "for(uint32_t i=0;i<1024;++i) { sum[i]=0.0f; compensation[i]=0.0f; }\n"
      << "for(uint32_t block=0;block<" << (k + 31) / 32 << "u;++block) {\n"
      << R"kernel(
    metadata.wait_front(1);
    auto scales=reinterpret_cast<volatile tt_l1_ptr float*>(metadata.get_read_ptr());
    for(uint32_t term=0;term<16;++term) {
      out.wait_front(1);
      auto tile=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out.get_read_ptr());
      float digit_weight=from_bits((127u-7u*(term/4+term%4))<<23);
      for(uint32_t row=0;row<32;++row) for(uint32_t col=0;col<32;++col) {
        uint32_t i=face_index(row,col), raw=tile[i];
        float partial=static_cast<float>(raw&0x7fffffffu);
        if(raw&0x80000000u) partial=-partial;
        partial=((partial*digit_weight)*scales[row])*scales[32+col];
        float correction=partial-compensation[i];
        float next=sum[i]+correction;
        compensation[i]=(next-sum[i])-correction;
        sum[i]=next;
      }
      out.pop_front(1);
    }
    metadata.pop_front(1);
    }
    for(uint32_t row=0;row<32;++row) for(uint32_t col=0;col<32;++col) {
      uint32_t m=(m_start+tm)*32+row, n=(n_start+tn)*32+col;
)kernel"
      << "if(m>=" << m << "u || n>=" << n << "u) continue;\n"
      << "float value=sum[face_index(row,col)];\n"
      // Exceptional exponent ranges are evaluated on device, never on host.
      << "if((to_bits(value)&0x7fffffffu)>0x7f800000u) {\nvalue=0.0f;\n"
      << "for(uint32_t k=0;k<" << k << "u;++k) {\n";
  if (convolution)
    writer << "uint32_t ic=k/" << ep.conv2d.kernel_h * ep.conv2d.kernel_w
           << "u, kh=(k/" << ep.conv2d.kernel_w << "u)%" << ep.conv2d.kernel_h
           << "u, kw=k%" << ep.conv2d.kernel_w << "u;\n";
  writer << "value += read_0(scratch," << lhsCoords << ") * read_1(scratch,"
         << rhsCoords << ");\n}\n}\n"
         << epilogue(ep) << "write_2(scratch,value," << coords(*t[2], "m", "n")
         << ");\n"
         << "}\n}\n}\n";
  result.writer = writer.str();
  result.compute = kCompute;
  return result;
}
TenstorrentNativeSources generateNativeSources(
    const tt_iree_ttex_entry_point_desc_t &ep, bool use_tensix) {
  return generateNativeSourcesImpl(ep, use_tensix);
}

std::string generateBF16NativeCommon(
    const tt_iree_ttex_tensor_desc_t *const *tensors) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV;
  auto source = commonSource(ep, tensors, TT_IREE_BF16_NATIVE_TENSOR_COUNT);
  const std::string old_comment = "native-v16 descriptor-specialized F32 storage ABI";
  source.replace(source.find(old_comment), old_comment.size(),
                 "native-v17 BF16/F32 storage and explicit IR rounding ABI");
  return "#pragma GCC optimize (\"no-fast-math\", \"fp-contract=off\")\n" + source;
}

TenstorrentNativeSources generateBF16NativeMatrix(
    const tt_iree_ttex_entry_point_desc_t &geometry,
    const tt_iree_ttex_tensor_desc_t *const *tensors,
    const std::string &epilogue, bool use_tensix) {
  if (use_tensix) {
    if (!tt_iree_ttex_is_native_matrix_kind(geometry.dispatch_kind))
      throw std::invalid_argument("BF16 Tensix codegen requires a matrix geometry");
    TenstorrentNativeSources sources;
    uint32_t m, n, k;
    std::string lhsCoords, rhsCoords;
    if (geometry.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION) {
      const auto &c = geometry.conv2d;
      m = c.output.shape[1] * c.output.shape[2];
      n = c.output.shape[0];
      k = c.input.shape[0] * c.kernel_h * c.kernel_w;
      lhsCoords = "ic,(m/" + std::to_string(c.output.shape[2]) + "u)*" +
          std::to_string(c.stride_h) + "u+kh*" + std::to_string(c.dilation_h) +
          "u,(m%" + std::to_string(c.output.shape[2]) + "u)*" +
          std::to_string(c.stride_w) + "u+kw*" + std::to_string(c.dilation_w) + "u";
      rhsCoords = "n,ic,kh,kw";
    } else if (geometry.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_LINEAR) {
      m = geometry.linear.m; n = geometry.linear.n; k = geometry.linear.k;
      lhsCoords = "m,k"; rhsCoords = "n,k";
    } else {
      m = geometry.matmul_like.m; n = geometry.matmul_like.n; k = geometry.matmul_like.k;
      lhsCoords = coords(*tensors[0], "m", "k"); rhsCoords = "n,k";
    }
    auto common = generateBF16NativeCommon(tensors) + bulkTransferSource(tensors);
    std::ostringstream reader;
    reader << common << R"kernel(
// native-v18: device tilization, BF16 operands, no host packing/fallback.
void kernel_main() {
  CircularBuffer lhs(tt::CBIndex::c_8), rhs(tt::CBIndex::c_9), scratch_cb(tt::CBIndex::c_2);
  CircularBuffer tile_a(tt::CBIndex::c_0), tile_b(tt::CBIndex::c_1);
  lhs.reserve_back(1); rhs.reserve_back(1); scratch_cb.reserve_back(1);
  uint32_t scratch=scratch_cb.get_write_ptr();
  auto a=reinterpret_cast<volatile tt_l1_ptr float*>(lhs.get_write_ptr());
  auto b=reinterpret_cast<volatile tt_l1_ptr float*>(rhs.get_write_ptr());
  uint32_t m_start=get_arg_val<uint32_t>(28), n_start=get_arg_val<uint32_t>(29);
  uint32_t m_count=get_arg_val<uint32_t>(30), n_count=get_arg_val<uint32_t>(31);
  for(uint32_t tm=0;tm<m_count;++tm) for(uint32_t tn=0;tn<n_count;++tn) {
    uint32_t m_base=(m_start+tm)*32,n_base=(n_start+tn)*32;
)kernel"
      << "uint32_t valid_m=" << m << "u-m_base; if(valid_m>32) valid_m=32;\n"
      << "uint32_t valid_n=" << n << "u-n_base; if(valid_n>32) valid_n=32;\n"
      << "for(uint32_t block=0;block<" << (k+31)/32 << "u;++block) {\n"
      << "uint32_t valid_k=" << k << "u-block*32; if(valid_k>32) valid_k=32;\n";
    emitNativeInputLoads(reader, geometry, tensors, lhsCoords, rhsCoords);
    reader << R"kernel(
    tile_a.reserve_back(1); tile_b.reserve_back(1);
    auto ta=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_a.get_write_ptr());
    auto tb=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_b.get_write_ptr());
    for(uint32_t row=0;row<32;++row) for(uint32_t col=0;col<32;++col) {
      ta[face_index(row,col)]=bf16_bits(a[row*32+col]);
      tb[face_index(row,col)]=bf16_bits(b[row*32+col]);
    }
    tile_a.push_back(1); tile_b.push_back(1);
  }
  }
}
)kernel";
    sources.reader = reader.str();
    sources.compute = R"kernel(
// native-v18: all K tiles accumulate in FP32 Dst before one final pack.
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
  uint32_t tiles=get_arg_val<uint32_t>(0), blocks=get_arg_val<uint32_t>(1);
  CircularBuffer lhs(tt::CBIndex::c_0), rhs(tt::CBIndex::c_1), out(tt::CBIndex::c_16);
  compute_kernel_hw_startup<SrcOrder::Reverse>(tt::CBIndex::c_0,tt::CBIndex::c_1,tt::CBIndex::c_16);
  matmul_init(tt::CBIndex::c_0,tt::CBIndex::c_1);
  for(uint32_t tile=0;tile<tiles;++tile) {
    tile_regs_acquire();
    for(uint32_t block=0;block<blocks;++block) {
      lhs.wait_front(1); rhs.wait_front(1);
      matmul_tiles(tt::CBIndex::c_0,tt::CBIndex::c_1,0,0,0);
      lhs.pop_front(1); rhs.pop_front(1);
    }
    tile_regs_commit(); tile_regs_wait();
    out.reserve_back(1); pack_tile(0,tt::CBIndex::c_16); out.push_back(1);
    tile_regs_release();
  }
}
)kernel";
    std::ostringstream writer;
    writer << common << R"kernel(
// Private writer scratch: never aliases the concurrently running reader.
void kernel_main() {
  CircularBuffer out(tt::CBIndex::c_16), accumulator(tt::CBIndex::c_17), scratch_cb(tt::CBIndex::c_3);
  accumulator.reserve_back(1); scratch_cb.reserve_back(1);
  uint32_t scratch=scratch_cb.get_write_ptr();
  auto acc=reinterpret_cast<volatile tt_l1_ptr float*>(accumulator.get_write_ptr());
  uint32_t m_start=get_arg_val<uint32_t>(28),n_start=get_arg_val<uint32_t>(29);
  uint32_t m_count=get_arg_val<uint32_t>(30),n_count=get_arg_val<uint32_t>(31);
  for(uint32_t tm=0;tm<m_count;++tm) for(uint32_t tn=0;tn<n_count;++tn) {
    uint32_t m_base=(m_start+tm)*32,n_base=(n_start+tn)*32;
)kernel"
      << "uint32_t valid_m=" << m << "u-m_base; if(valid_m>32) valid_m=32;\n"
      << "uint32_t valid_n=" << n << "u-n_base; if(valid_n>32) valid_n=32;\n"
      << "out.wait_front(1);\n"
      << "auto packed=reinterpret_cast<volatile tt_l1_ptr float*>(out.get_read_ptr());\n"
      << "for(uint32_t row=0;row<valid_m;++row) for(uint32_t col=0;col<valid_n;++col) {\n"
      << "uint32_t m=m_base+row,n=n_base+col; float value=packed[face_index(row,col)];\n"
      << epilogue << "acc[row*32+col]=value;\n}\n";
    emitNativeOutputStores(writer, *tensors[2]);
    writer << "out.pop_front(1);\n}\n}\n";
    sources.writer = writer.str();
    return sources;
  }
  auto sources = generateNativeSourcesImpl(geometry, false, tensors, &epilogue);
  const std::string old_comment = "native-v15 descriptor-specialized F32 storage ABI";
  auto comment = sources.reader.find(old_comment);
  if (comment != std::string::npos)
    sources.reader.replace(comment, old_comment.size(), "native-v17 BF16/F32 storage and explicit IR rounding ABI");
  // v17 extends only this dispatch's address/base list, not legacy v15/v16.
  for (unsigned i = 20; i < 24; ++i) {
    std::string from = "get_arg_val<uint32_t>(" + std::to_string(i) + ")";
    std::string to = "get_arg_val<uint32_t>(" + std::to_string(i + 8) + ")";
    size_t pos = sources.reader.find("void kernel_main()");
    while ((pos = sources.reader.find(from, pos)) != std::string::npos) {
      sources.reader.replace(pos, from.size(), to);
      pos += to.size();
    }
  }
  return sources;
}
} // namespace mlir::iree_compiler::IREE::HAL
