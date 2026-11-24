#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

static const std::vector<std::string> TOKENS = {
    "    <revision>\n      <id>",
    "</id>\n      <timestamp>",
    "    <timestamp>",
    "</timestamp>\n      <contributor>\n        <username>",
    "      <username>",
    "</username>\n        <id>",
    "        <id>",
    "        </id>\n      </contributor>\n",
    "      <comment>",
    "      </comment>\n",
    "      <text xml:space=\"preserve\">",
    "      </text>\n    </revision>\n  </page>\n  <page>\n    <title>",
    "    <title>",
    "</title>\n    <id>",
    "      <minor />\n",
    "      <ip>",
    "</ip>\n      </contributor>\n",
    "&quot;", "&amp;", "&lt;", "&gt;", "&apos;", "&ndash;", "&mdash;",
    "[[Category:", "[[Image:", "#REDIRECT [[", "http://www.", "https://",
    "==External links==", "==See also==", "==References==", "{{cite ",
    " the ", " of ", " and ", " in ", " to ", " a ", " is ", " was ",
    " for ", " that ", " on ", " with ", " as ", " by ", " at ", " from ",
    " are ", " be ", " this ", " which ", " or ", " had ", " were ", " but ",
    " not ", " have ", " an ", " they ", " one ", " all ", " their ", " there ",
    " been ", " has ", " more ", " when ", " who ", " will ", " would ", " into ",
    " than ", " other ", " its ", " about ", " after ", " first ", " also ", " new ",
    " because ", " two ", " these ", " time ", " can ", " only ", " between ", " most ",
    " such ", " where ", " through ", " years ", " during ", " before ", " under ", " states ",
    " united ", " american ", " world ", " university ", " national ", " century ", " history ", " government ",
    " population ", " language ", " system ", " music ", " war ", " city ", " state ", " people ",
    "tion ", "ment ", "ence ", "ance ", "able ", "ible ", "ing ", "ed ", "ly ", "er ",
    "The ", "In ", "It ", "He ", "As ", "On ", "For ", "By ", "With ", "From ",
    "'''", "''", "[[", "]]", "{{", "}}", "==", "==="
};

void encode_file(const std::string& in_path, const std::string& out_path) {
    std::ifstream in(in_path, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    std::vector<std::vector<size_t>> by_first(256);
    for (size_t i = 0; i < TOKENS.size(); ++i) {
        by_first[(uint8_t)TOKENS[i][0]].push_back(i);
    }

    std::ofstream out(out_path, std::ios::binary);
    size_t n = data.size();
    size_t i = 0;
    while (i < n) {
        uint8_t c = (uint8_t)data[i];
        int matched_tok = -1;

        for (size_t tok_id : by_first[c]) {
            const std::string& tok = TOKENS[tok_id];
            if (i + tok.size() <= n && data.compare(i, tok.size(), tok) == 0) {
                matched_tok = (int)tok_id;
                break;
            }
        }

        if (matched_tok >= 0) {
            out.put((char)0xFE);
            out.put((char)(uint8_t)matched_tok);
            i += TOKENS[matched_tok].size();
        } else if (c == 0xFE || c == 0xFF) {
            out.put((char)0xFF);
            out.put((char)c);
            i++;
        } else {
            out.put((char)c);
            i++;
        }
    }
}

void decode_file(const std::string& in_path, const std::string& out_path) {
    std::ifstream in(in_path, std::ios::binary);
    std::ofstream out(out_path, std::ios::binary);

    int c;
    while ((c = in.get()) != EOF) {
        uint8_t b = (uint8_t)c;
        if (b == 0xFE) {
            int tok_id = in.get();
            if (tok_id != EOF && (size_t)tok_id < TOKENS.size()) {
                out.write(TOKENS[tok_id].data(), TOKENS[tok_id].size());
            }
        } else if (b == 0xFF) {
            int lit = in.get();
            if (lit != EOF) out.put((char)(uint8_t)lit);
        } else {
            out.put((char)b);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./wiki_prep [e|d] <input> <output>\n";
        return 1;
    }
    if (argv[1][0] == 'e') encode_file(argv[2], argv[3]);
    else                  decode_file(argv[2], argv[3]);
    return 0;
}
