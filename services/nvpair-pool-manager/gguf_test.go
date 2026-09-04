// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"nvpair-shared/vrampool"
)

// ggufBuilder writes a synthetic GGUF header. Synthesizing one is far better
// than checking in a fixture: a real model file is gigabytes, and a truncated
// one would not exercise the tensor table this reader depends on.
type ggufBuilder struct {
	buf     bytes.Buffer
	kv      bytes.Buffer
	kvCount uint64
	tensors bytes.Buffer
	nTensor uint64
}

func (b *ggufBuilder) putString(w *bytes.Buffer, s string) {
	_ = binary.Write(w, binary.LittleEndian, uint64(len(s)))
	w.WriteString(s)
}

func (b *ggufBuilder) stringKV(key, value string) {
	b.putString(&b.kv, key)
	_ = binary.Write(&b.kv, binary.LittleEndian, ggufString)
	b.putString(&b.kv, value)
	b.kvCount++
}

func (b *ggufBuilder) uint32KV(key string, value uint32) {
	b.putString(&b.kv, key)
	_ = binary.Write(&b.kv, binary.LittleEndian, ggufUint32)
	_ = binary.Write(&b.kv, binary.LittleEndian, value)
	b.kvCount++
}

// arrayKV writes an array-valued entry, which the reader must skip cleanly —
// real models carry a tokenizer vocabulary this way, and mis-skipping it would
// desynchronize everything after it.
func (b *ggufBuilder) arrayKV(key string, values []uint32) {
	b.putString(&b.kv, key)
	_ = binary.Write(&b.kv, binary.LittleEndian, ggufArray)
	_ = binary.Write(&b.kv, binary.LittleEndian, ggufUint32)
	_ = binary.Write(&b.kv, binary.LittleEndian, uint64(len(values)))
	for _, v := range values {
		_ = binary.Write(&b.kv, binary.LittleEndian, v)
	}
	b.kvCount++
}

func (b *ggufBuilder) tensor(name string, offset uint64) {
	b.putString(&b.tensors, name)
	_ = binary.Write(&b.tensors, binary.LittleEndian, uint32(1)) // one dimension
	_ = binary.Write(&b.tensors, binary.LittleEndian, uint64(1)) // shape
	_ = binary.Write(&b.tensors, binary.LittleEndian, uint32(0)) // ggml type f32
	_ = binary.Write(&b.tensors, binary.LittleEndian, offset)
	b.nTensor++
}

// write assembles the file and pads it out with dataBytes of tensor data, since
// the reader derives sizes from the file's length.
func (b *ggufBuilder) write(t *testing.T, path string, dataBytes int) {
	t.Helper()
	_ = binary.Write(&b.buf, binary.LittleEndian, uint32(ggufMagic))
	_ = binary.Write(&b.buf, binary.LittleEndian, uint32(3))
	_ = binary.Write(&b.buf, binary.LittleEndian, b.nTensor)
	_ = binary.Write(&b.buf, binary.LittleEndian, b.kvCount)
	b.buf.Write(b.kv.Bytes())
	b.buf.Write(b.tensors.Bytes())
	b.buf.Write(make([]byte, dataBytes))
	if err := os.WriteFile(path, b.buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
}

// llamaModel builds a small but structurally realistic model: four repeating
// blocks plus embedding and output tensors, with grouped-query attention.
func llamaModel(t *testing.T, dir string) string {
	t.Helper()
	b := &ggufBuilder{}
	b.stringKV("general.architecture", "llama")
	b.stringKV("general.name", "test")
	b.uint32KV("llama.block_count", 4)
	b.uint32KV("llama.embedding_length", 4096)
	b.uint32KV("llama.attention.head_count", 32)
	b.uint32KV("llama.attention.head_count_kv", 8)
	b.arrayKV("tokenizer.ggml.token_type", []uint32{1, 2, 3, 4, 5})

	// 1 MiB per tensor, laid out contiguously the way GGUF requires.
	const each = 1 << 20
	b.tensor("token_embd.weight", 0)
	b.tensor("blk.0.attn_q.weight", each)
	b.tensor("blk.1.attn_q.weight", 2*each)
	b.tensor("blk.2.attn_q.weight", 3*each)
	b.tensor("blk.3.attn_q.weight", 4*each)
	b.tensor("output.weight", 5*each)

	path := filepath.Join(dir, "test-model.gguf")
	b.write(t, path, 6*each)
	return path
}

func TestReadModelParsesAGGUFHeader(t *testing.T) {
	path := llamaModel(t, t.TempDir())
	got, err := ReadModel(path)
	if err != nil {
		t.Fatalf("ReadModel: %v", err)
	}

	if got.Name != "test-model" {
		t.Errorf("Name = %q, want the file's stem", got.Name)
	}
	if got.Layers != 4 {
		t.Errorf("Layers = %d, want 4", got.Layers)
	}
	if got.WeightBytes == 0 {
		t.Error("WeightBytes should be the file size")
	}
	if err := got.Validate(); err != nil {
		t.Errorf("the parsed model is not plannable: %v", err)
	}
}

// The grouped width, not the embedding width. On this model they differ by a
// factor of four, which is the difference between a plan that fits and one that
// never could.
func TestKVSizeUsesTheGroupedWidth(t *testing.T) {
	path := llamaModel(t, t.TempDir())
	got, err := ReadModel(path)
	if err != nil {
		t.Fatalf("ReadModel: %v", err)
	}
	// head_dim = 4096/32 = 128; kv width = 128 * 8 = 1024; times 2 for K and V,
	// times 2 bytes an element.
	if want := vrampool.KVBytesPerLayerPerToken(1024, 2); got.KVBytesPerLayerPerToken != want {
		t.Fatalf("KVBytesPerLayerPerToken = %d, want %d", got.KVBytesPerLayerPerToken, want)
	}
}

// A model with no grouped-query metadata has one KV head per attention head.
func TestKVSizeFallsBackToUngroupedAttention(t *testing.T) {
	dir := t.TempDir()
	b := &ggufBuilder{}
	b.stringKV("general.architecture", "gptneox")
	b.uint32KV("gptneox.block_count", 2)
	b.uint32KV("gptneox.embedding_length", 2048)
	b.uint32KV("gptneox.attention.head_count", 16)
	b.tensor("token_embd.weight", 0)
	b.tensor("blk.0.attn_q.weight", 1<<20)
	path := filepath.Join(dir, "m.gguf")
	b.write(t, path, 2<<20)

	got, err := ReadModel(path)
	if err != nil {
		t.Fatalf("ReadModel: %v", err)
	}
	if want := vrampool.KVBytesPerLayerPerToken(2048, 2); got.KVBytesPerLayerPerToken != want {
		t.Fatalf("KVBytesPerLayerPerToken = %d, want the full embedding width %d",
			got.KVBytesPerLayerPerToken, want)
	}
}

// The head node carries the embedding and output tensors, so they must be
// separated from the repeating blocks — charging them per layer would make every
// device look bigger than it is.
func TestNonRepeatingTensorsAreSeparated(t *testing.T) {
	path := llamaModel(t, t.TempDir())
	got, err := ReadModel(path)
	if err != nil {
		t.Fatalf("ReadModel: %v", err)
	}
	// token_embd and output, one MiB each.
	const each = uint64(1) << 20
	if got.NonRepeatingBytes != 2*each {
		t.Fatalf("NonRepeatingBytes = %d, want %d", got.NonRepeatingBytes, 2*each)
	}
	// Four blocks share what is left.
	if perLayer := got.PerLayerWeightBytes(); perLayer != each {
		t.Fatalf("PerLayerWeightBytes = %d, want %d", perLayer, each)
	}
}

func TestReadModelRejectsWhatItCannotPlanAgainst(t *testing.T) {
	dir := t.TempDir()

	notGGUF := filepath.Join(dir, "not.gguf")
	if err := os.WriteFile(notGGUF, []byte("this is a text file"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadModel(notGGUF); err == nil {
		t.Error("a non-GGUF file should be refused")
	}

	if _, err := ReadModel(filepath.Join(dir, "absent.gguf")); err == nil {
		t.Error("a missing file should be refused")
	}

	// A header that declares an architecture but no block count cannot be
	// planned against, and guessing a layer count would produce a plan that
	// looks authoritative and is wrong.
	b := &ggufBuilder{}
	b.stringKV("general.architecture", "llama")
	noLayers := filepath.Join(dir, "nolayers.gguf")
	b.write(t, noLayers, 1024)
	if _, err := ReadModel(noLayers); err == nil {
		t.Error("a model with no block count should be refused")
	}

	// No architecture at all: every other key is namespaced by it, so nothing
	// can be looked up.
	c := &ggufBuilder{}
	c.uint32KV("llama.block_count", 4)
	noArch := filepath.Join(dir, "noarch.gguf")
	c.write(t, noArch, 1024)
	if _, err := ReadModel(noArch); err == nil {
		t.Error("a model with no architecture should be refused")
	}
}

// The counts are read from the file, so allocating from them unchecked is how a
// malformed header becomes an out-of-memory.
func TestReadModelRefusesImplausibleCounts(t *testing.T) {
	dir := t.TempDir()
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint32(ggufMagic))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(3))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(1)<<40) // tensor count
	_ = binary.Write(&buf, binary.LittleEndian, uint64(1))
	path := filepath.Join(dir, "huge.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadModel(path); err == nil {
		t.Fatal("an implausible tensor count should be refused")
	}
}

func TestReadModelRejectsAnUnsupportedVersion(t *testing.T) {
	dir := t.TempDir()
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint32(ggufMagic))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(99))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(0))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(0))
	path := filepath.Join(dir, "future.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadModel(path); err == nil {
		t.Fatal("an unsupported GGUF version should be refused rather than guessed at")
	}
}
