package model

import (
	"fmt"
	"os"
	"reflect"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/ml/nn"
)

func TestParseTags(t *testing.T) {
	cases := []struct {
		value string
		want  Tag
	}{
		{
			value: "output",
			want: Tag{
				Name: "output",
			},
		},
		{
			value: "output,alt:token_embd",
			want: Tag{
				Name: "output",
				Alternate: []string{
					"token_embd",
				},
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.value, func(t *testing.T) {
			got := ParseTags(tt.value)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ParseTags() returned unexpected values (-want +got):\n%s", diff)
			}
		})
	}
}

type fakeBackend struct {
	*ggml.Backend
	names []string
}

type fakeTensor struct {
	*ggml.Tensor
	Name string
}

func (m *fakeBackend) Get(name string) ml.Tensor {
	if slices.Contains(m.names, name) {
		return &fakeTensor{Name: name}
	}

	return nil
}

func TestPopulateFields(t *testing.T) {
	type fakeLayer struct {
		Query  *nn.Linear `gguf:"attn_q"`
		Key    *nn.Linear `gguf:"attn_k"`
		Value  *nn.Linear `gguf:"attn_v"`
		Output *nn.Linear `gguf:"attn_o"`
	}

	type fakeModel struct {
		Input      *nn.Embedding `gguf:"input"`
		OutputNorm *nn.RMSNorm   `gguf:"output_norm"`
		Output     *nn.Linear    `gguf:"output"`
		Layers     [2]fakeLayer  `gguf:"blk"`
	}

	var m fakeModel
	v := reflect.ValueOf(&m)
	v.Elem().Set(populateFields(Base{b: &fakeBackend{
		names: []string{
			"input.weight",
			"blk.0.attn_q.weight",
			"blk.0.attn_k.weight",
			"blk.0.attn_v.weight",
			"blk.1.attn_q.weight",
			"blk.1.attn_k.weight",
			"blk.1.attn_v.weight",
			"output_norm.weight",
			"output.weight",
		},
	}}, v.Elem()))

	if diff := cmp.Diff(fakeModel{
		Input:      &nn.Embedding{Weight: &fakeTensor{Name: "input.weight"}},
		OutputNorm: &nn.RMSNorm{Weight: &fakeTensor{Name: "output_norm.weight"}},
		Output:     &nn.Linear{Weight: &fakeTensor{Name: "output.weight"}},
		Layers: [2]fakeLayer{
			{
				Query: &nn.Linear{Weight: &fakeTensor{Name: "blk.0.attn_q.weight"}},
				Key:   &nn.Linear{Weight: &fakeTensor{Name: "blk.0.attn_k.weight"}},
				Value: &nn.Linear{Weight: &fakeTensor{Name: "blk.0.attn_v.weight"}},
			},
			{
				Query: &nn.Linear{Weight: &fakeTensor{Name: "blk.1.attn_q.weight"}},
				Key:   &nn.Linear{Weight: &fakeTensor{Name: "blk.1.attn_k.weight"}},
				Value: &nn.Linear{Weight: &fakeTensor{Name: "blk.1.attn_v.weight"}},
			},
		},
	}, m); diff != "" {
		t.Errorf("populateFields() set incorrect values (-want +got):\n%s", diff)
	}
}

func TestPopulateFieldsAlternateName(t *testing.T) {
	type fakeModel struct {
		Input  *nn.Embedding `gguf:"input"`
		Output *nn.Linear    `gguf:"output,alt:input"`
	}

	m := fakeModel{}
	v := reflect.ValueOf(&m)
	v.Elem().Set(populateFields(Base{b: &fakeBackend{
		names: []string{
			"input.weight",
		},
	}}, v.Elem()))

	if diff := cmp.Diff(fakeModel{
		Input:  &nn.Embedding{Weight: &fakeTensor{Name: "input.weight"}},
		Output: &nn.Linear{Weight: &fakeTensor{Name: "input.weight"}},
	}, m); diff != "" {
		t.Errorf("populateFields() set incorrect values (-want +got):\n%s", diff)
	}
}

func TestForwardSimple(t *testing.T) {
	p := "../convert/testdata/Meta-Llama-3-8B-Instruct"
	if testing.Short() {
		t.Skip("skipping in short mode")
	} else if _, err := os.Stat(p); err != nil {
		t.Skipf("%s not found", p)
	}

	m, err := New(p)
	if err != nil {
		t.Fatal(err)
	}

	m.Config().Cache.Init(m.Backend(), ml.DTypeF32, 2048)

	// Create inputs from text
	n, err := m.(TextProcessor).Encode("hi")
	if err != nil {
		t.Fatal(err)
	}

	// Setup options based on sequence pattern
	options := Options{
		Inputs:    n,
		Positions: make([]int32, len(n)),
		Sequences: make([]int, len(n)),
		Outputs:   []int32{int32(len(n) - 1)},
	}
	for i := range options.Positions {
		options.Positions[i] = int32(i)
		options.Sequences[i] = 1
	}

	ctx := m.Backend().NewContext()
	defer ctx.Close()

	modelOutput, err := Forward(ctx, m, options)
	if err != nil {
		t.Fatal(fmt.Errorf("forward pass failed: %v", err))
	}
	for i := range options.Positions {
		options.Positions[i] = int32(i)
		options.Sequences[i] = 1
	}

	// Verify the output is populated
	if modelOutput == nil {
		t.Error("expected non-nil model output")
	}

	// Compute the output
	ctx.Compute(modelOutput)
}
