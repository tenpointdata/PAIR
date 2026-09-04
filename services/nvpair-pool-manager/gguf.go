// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"nvpair-shared/vrampool"
)

// GGUF header constants. The format is little-endian throughout, versioned, and
// self-describing; only the parts a capacity plan needs are read here.
const (
	ggufMagic   = 0x46554747 // "GGUF"
	ggufMinVer  = 2
	ggufMaxVer  = 3
	ggufMaxKeys = 1 << 16
	// ggufMaxTensors bounds the tensor table. A 405B model has a few thousand
	// tensors; the cap exists because the count is read from the file and
	// allocating from it unchecked is how a malformed header becomes an
	// out-of-memory.
	ggufMaxTensors = 1 << 20
	// ggufMaxStringLen bounds one metadata string, for the same reason.
	ggufMaxStringLen = 1 << 20
	// ggufMaxArrayLen bounds an array-valued metadata entry.
	ggufMaxArrayLen = 1 << 22
)

// GGUF metadata value types.
const (
	ggufUint8 uint32 = iota
	ggufInt8
	ggufUint16
	ggufInt16
	ggufUint32
	ggufInt32
	ggufFloat32
	ggufBool
	ggufString
	ggufArray
	ggufUint64
	ggufInt64
	ggufFloat64
)

// defaultKVElementBytes is the per-element size of the key-value cache. f16 is
// llama.cpp's default; a quantized cache is smaller, so assuming f16 plans for
// the larger case, which is the safe direction for a capacity estimate.
const defaultKVElementBytes = 2

// blockTensorPrefix marks a tensor belonging to one repeating layer. Everything
// without it — the token embedding, the output norm, the output head — is
// charged to the pool's head node, which is where llama.cpp keeps it.
const blockTensorPrefix = "blk."

// ggufTensor is one entry of the tensor table, reduced to what a size estimate
// needs.
type ggufTensor struct {
	name   string
	offset uint64
}

// ReadModel reads a GGUF file's header and returns the model description the
// planner needs.
//
// It reads the header only — never a tensor — so it answers "does this fit"
// without touching gigabytes, which is the whole point of asking before a pool
// is formed. Total weight size comes from the file's length rather than from
// summing tensors, for the same reason the per-tensor sizes below do not decode
// quantization types.
//
// Tensor sizes are derived from the DIFFERENCES BETWEEN OFFSETS rather than from
// each tensor's dimensions and quantization type. GGUF lays tensor data out
// contiguously in table order, so the gap to the next offset is that tensor's
// size plus any alignment padding — which makes the estimate very slightly high,
// and high is the safe direction. The alternative is a table mapping every ggml
// quantization type to its block layout, which is a second copy of upstream data
// that would go stale the first time a new type is added.
func ReadModel(path string) (vrampool.Model, error) {
	var out vrampool.Model

	f, err := os.Open(path)
	if err != nil {
		return out, fmt.Errorf("open model: %w", err)
	}
	defer func() { _ = f.Close() }()

	info, err := f.Stat()
	if err != nil {
		return out, fmt.Errorf("stat model: %w", err)
	}
	fileSize := uint64(info.Size())

	r := &ggufReader{r: f}

	magic, err := r.u32()
	if err != nil || magic != ggufMagic {
		return out, errors.New("not a GGUF file")
	}
	version, err := r.u32()
	if err != nil {
		return out, fmt.Errorf("read version: %w", err)
	}
	if version < ggufMinVer || version > ggufMaxVer {
		return out, fmt.Errorf("unsupported GGUF version %d", version)
	}
	tensorCount, err := r.u64()
	if err != nil {
		return out, fmt.Errorf("read tensor count: %w", err)
	}
	if tensorCount > ggufMaxTensors {
		return out, fmt.Errorf("implausible tensor count %d", tensorCount)
	}
	kvCount, err := r.u64()
	if err != nil {
		return out, fmt.Errorf("read metadata count: %w", err)
	}
	if kvCount > ggufMaxKeys {
		return out, fmt.Errorf("implausible metadata count %d", kvCount)
	}

	meta := make(map[string]uint64, kvCount)
	var arch string
	for range kvCount {
		key, err := r.str()
		if err != nil {
			return out, fmt.Errorf("read metadata key: %w", err)
		}
		valueType, err := r.u32()
		if err != nil {
			return out, fmt.Errorf("read type of %q: %w", key, err)
		}
		if key == "general.architecture" && valueType == ggufString {
			s, err := r.str()
			if err != nil {
				return out, fmt.Errorf("read architecture: %w", err)
			}
			arch = s
			continue
		}
		// Only unsigned integer metadata matters here (layer counts and widths).
		// Everything else is skipped by length rather than decoded, so an
		// unfamiliar value type costs nothing.
		if v, ok, err := r.uintValue(valueType); err != nil {
			return out, fmt.Errorf("read value of %q: %w", key, err)
		} else if ok {
			meta[key] = v
		}
	}

	tensors := make([]ggufTensor, 0, tensorCount)
	for range tensorCount {
		name, err := r.str()
		if err != nil {
			return out, fmt.Errorf("read tensor name: %w", err)
		}
		dims, err := r.u32()
		if err != nil {
			return out, fmt.Errorf("read dims of %q: %w", name, err)
		}
		if dims > 8 {
			return out, fmt.Errorf("tensor %q claims %d dimensions", name, dims)
		}
		for range dims {
			if _, err := r.u64(); err != nil {
				return out, fmt.Errorf("read shape of %q: %w", name, err)
			}
		}
		if _, err := r.u32(); err != nil { // ggml type
			return out, fmt.Errorf("read type of %q: %w", name, err)
		}
		offset, err := r.u64()
		if err != nil {
			return out, fmt.Errorf("read offset of %q: %w", name, err)
		}
		tensors = append(tensors, ggufTensor{name: name, offset: offset})
	}

	if arch == "" {
		return out, errors.New("model declares no architecture")
	}
	layers := meta[arch+".block_count"]
	if layers == 0 {
		return out, fmt.Errorf("model declares no %s.block_count", arch)
	}

	out.Name = strings.TrimSuffix(info.Name(), ".gguf")
	out.Layers = int(layers)
	// The tensor data, not the whole file: the header is metadata and a
	// vocabulary, which never reach a GPU. On a multi-gigabyte model the
	// difference is immaterial, but charging it as weights would still be
	// charging the wrong thing.
	out.WeightBytes = fileSize - min(r.offset, fileSize)
	out.NonRepeatingBytes = nonRepeatingBytes(tensors, r.offset, fileSize)
	out.KVBytesPerLayerPerToken = vrampool.KVBytesPerLayerPerToken(kvWidth(meta, arch), defaultKVElementBytes)
	return out, nil
}

// kvWidth derives the grouped key-value projection width, which is what the
// cache is actually sized by.
//
// The grouped width, not the embedding width. On a grouped-query model the two
// differ by the head ratio — commonly a factor of four or eight — and using the
// embedding width would overstate the cache by exactly that factor, which is the
// difference between a plan that fits and one that never could.
func kvWidth(meta map[string]uint64, arch string) uint64 {
	embedding := meta[arch+".embedding_length"]
	heads := meta[arch+".attention.head_count"]
	kvHeads := meta[arch+".attention.head_count_kv"]

	if embedding == 0 || heads == 0 {
		return embedding
	}
	if kvHeads == 0 {
		// No grouped-query metadata means every head has its own KV.
		kvHeads = heads
	}
	headDim := embedding / heads
	return headDim * kvHeads
}

// nonRepeatingBytes sums the tensors that are not part of a repeating block.
func nonRepeatingBytes(tensors []ggufTensor, dataStart, fileSize uint64) uint64 {
	if len(tensors) == 0 || fileSize <= dataStart {
		return 0
	}
	sorted := make([]ggufTensor, len(tensors))
	copy(sorted, tensors)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].offset < sorted[j].offset })

	dataSize := fileSize - dataStart
	var total uint64
	for i, t := range sorted {
		end := dataSize
		if i+1 < len(sorted) {
			end = sorted[i+1].offset
		}
		if end < t.offset {
			continue
		}
		if !strings.HasPrefix(t.name, blockTensorPrefix) {
			total += end - t.offset
		}
	}
	if total > fileSize {
		return fileSize
	}
	return total
}

// ggufReader reads the little-endian primitives the header is built from,
// tracking how far it has read so the tensor-data offset is known.
type ggufReader struct {
	r      io.Reader
	offset uint64
	buf    [8]byte
}

func (g *ggufReader) read(n int) ([]byte, error) {
	if _, err := io.ReadFull(g.r, g.buf[:n]); err != nil {
		return nil, err
	}
	g.offset += uint64(n)
	return g.buf[:n], nil
}

func (g *ggufReader) u32() (uint32, error) {
	b, err := g.read(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (g *ggufReader) u64() (uint64, error) {
	b, err := g.read(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

func (g *ggufReader) str() (string, error) {
	n, err := g.u64()
	if err != nil {
		return "", err
	}
	if n > ggufMaxStringLen {
		return "", fmt.Errorf("implausible string length %d", n)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(g.r, buf); err != nil {
		return "", err
	}
	g.offset += n
	return string(buf), nil
}

// uintValue decodes a metadata value when it is an unsigned integer, and
// otherwise skips it. ok is false for a value that was skipped.
func (g *ggufReader) uintValue(valueType uint32) (uint64, bool, error) {
	switch valueType {
	case ggufUint8, ggufInt8, ggufBool:
		b, err := g.read(1)
		if err != nil {
			return 0, false, err
		}
		return uint64(b[0]), true, nil
	case ggufUint16, ggufInt16:
		b, err := g.read(2)
		if err != nil {
			return 0, false, err
		}
		return uint64(binary.LittleEndian.Uint16(b)), true, nil
	case ggufUint32, ggufInt32, ggufFloat32:
		v, err := g.u32()
		if err != nil {
			return 0, false, err
		}
		if valueType == ggufFloat32 {
			return 0, false, nil
		}
		return uint64(v), true, nil
	case ggufUint64, ggufInt64, ggufFloat64:
		v, err := g.u64()
		if err != nil {
			return 0, false, err
		}
		if valueType == ggufFloat64 {
			return 0, false, nil
		}
		return v, true, nil
	case ggufString:
		if _, err := g.str(); err != nil {
			return 0, false, err
		}
		return 0, false, nil
	case ggufArray:
		elemType, err := g.u32()
		if err != nil {
			return 0, false, err
		}
		n, err := g.u64()
		if err != nil {
			return 0, false, err
		}
		if n > ggufMaxArrayLen {
			return 0, false, fmt.Errorf("implausible array length %d", n)
		}
		for range n {
			if _, _, err := g.uintValue(elemType); err != nil {
				return 0, false, err
			}
		}
		return 0, false, nil
	default:
		return 0, false, fmt.Errorf("unknown metadata type %d", valueType)
	}
}
