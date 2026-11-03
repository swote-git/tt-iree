// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_TENSTORRENTENCODING_H_
#define TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_TENSTORRENTENCODING_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler::tenstorrent {

// Data-tiling layout for Tenstorrent: BF16 matmul operands become 32x32 tiles
// whose 16x16 faces are stored in Tensix order, so a dispatch-boundary tile is
// byte-identical to a circular-buffer tile.
//
// LHS [M,K]:    [M1,K1, mf,kf, 16,16]
// RHS [K,N]:    [N1,K1, kf,nf, 16,16]   (K-major tile, no unpacker transpose)
// RESULT [M,N]: [M1,N1, mf,nf, 16,16]
//
// Contractions over these operands lower to a linalg.generic carrying
// kTileMatmulAttrName.
inline constexpr int64_t kTileSize = 32;
inline constexpr int64_t kFaceSize = 16;
inline constexpr char kTileMatmulAttrName[] = "tt_iree.tile_matmul";

namespace detail {
struct EncodingResolverAttrStorage;
} // namespace detail

// #tt_iree.encoding_resolver<{configuration}>
class EncodingResolverAttr
    : public Attribute::AttrBase<EncodingResolverAttr, Attribute,
                                 detail::EncodingResolverAttrStorage> {
public:
  using Base::Base;
  static constexpr StringLiteral name = "tt_iree.encoding_resolver";
  static EncodingResolverAttr get(MLIRContext *context,
                                  DictionaryAttr configuration);
  DictionaryAttr getConfiguration() const;
};

class TTIREEDialect : public Dialect {
public:
  explicit TTIREEDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tt_iree"; }
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  void printAttribute(Attribute attr, DialectAsmPrinter &printer) const override;
};

// Registers the dialect and attaches the IREE encoding interfaces.
void registerTenstorrentEncoding(DialectRegistry &registry);

} // namespace mlir::iree_compiler::tenstorrent

MLIR_DECLARE_EXPLICIT_TYPE_ID(
    mlir::iree_compiler::tenstorrent::EncodingResolverAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::iree_compiler::tenstorrent::TTIREEDialect)

#endif // TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_TENSTORRENTENCODING_H_
