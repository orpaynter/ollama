/**
 * llama.cpp - git 059031b8c40e1f4ba60586842c5b1ed3ddf61842
 *
 * MIT License
 *
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Implements a parser for an extended Backus-Naur form (BNF), producing the
// binary context-free grammar format specified by llama.h. Supports character
// ranges, grouping, and repetition operators. As an example, a grammar for
// arithmetic might look like:
//
// root  ::= expr
// expr  ::= term ([-+*/] term)*
// term  ::= num | "(" space expr ")" space
// num   ::= [0-9]+ space
// space ::= [ \t\n]*

#pragma once
#include "llama.h"
#include <vector>
#include <map>
#include <cstdint>
#include <string>

namespace grammar_parser {
    struct parse_state {
        std::map<std::string, uint32_t>                 symbol_ids;
        std::vector<std::vector<llama_grammar_element>> rules;

        std::vector<const llama_grammar_element *> c_rules();
    };

    parse_state parse(const char * src);
    void print_grammar(FILE * file, const parse_state & state);
}
