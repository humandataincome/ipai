import * as ipdw from "ipdw";

type Child = {
    [key: string]: TrieNode;
}

class TrieNode {
    key: string | undefined;
    parent?: TrieNode;
    children: Child;
    end: boolean;
    score?: number;
    index?: number;

    constructor(key: string | undefined) {
        this.key = key;
        this.parent = undefined;
        this.children = {};
        this.end = false;
    }

    getWord() {
        const output: string[] = [];
        let node: TrieNode | undefined;
        while (node !== undefined) {
            if (node.key !== undefined) {
                output.unshift(node.key);
            }
            node = node.parent;
        }
        return [output, this.score, this.index];
    }
}

class Trie {
    root: TrieNode;

    constructor() {
        this.root = new TrieNode(undefined);
    }

    insert(word: string, score: number, index: number) {
        let node = this.root;

        const symbols: string[] = [];
        for (const symbol of word) {
            symbols.push(symbol);
        }

        for (let i: number = 0; i < symbols.length; i++) {
            if (!node.children[symbols[i]]) {
                node.children[symbols[i]] = new TrieNode(symbols[i]);
                node.children[symbols[i]].parent = node;
            }
            node = node.children[symbols[i]];
            if (i === symbols.length - 1) {
                node.end = true;
                node.score = score;
                node.index = index;
            }
        }
    }

    find(ss: string) {
        let node = this.root;
        let iter = 0;

        while (iter < ss.length && node != undefined) {
            node = node.children[ss[iter]];
            iter++;
        }

        return node;
    }
}

export class BertTokenizer {
    separator: string;
    UNK_INDEX: number;
    trie?: Trie;
    vocab?: string[];

    constructor(vocab: string) {
        this.separator = '\u2581';
        this.UNK_INDEX = 100;
        this.vocab = JSON.parse(vocab);

        this.trie = new Trie();
        this.trie.insert('[CLS]', 1, 101);
        this.trie.insert('[SEP]', 1, 102);

        // Actual tokens start at 999.
        for (let i = 999; i < this.vocab!.length; i++) {
            const word = this.vocab![i];
            this.trie.insert(word, 1, i);
        }
    }

    public static async Load(vocabPath: string, progress?: ((progress: number) => void) | undefined): Promise<BertTokenizer> {
        const persistence = await ipdw.Persistence.getInstance();
        const vocab = await persistence.fetchOrGet(vocabPath, progress ? p => progress(p) : undefined);

        return new BertTokenizer(new TextDecoder().decode(vocab));
    }

    processInput(text: string) {
        const words = text.split(' ');
        return words.map(word => {
            if (word !== '[CLS]' && word !== '[SEP]') {
                return this.separator + word.toLowerCase().normalize('NFKC');
            }
            return word;
        });
    }

    tokenize(text: string) {
        // Source:
        // https://github.com/google-research/bert/blob/88a817c37f788702a363ff935fd173b6dc6ac0d6/tokenization.py#L311
        let outputTokens: number[] = [];
        const words = this.processInput(text);

        for (let i = 0; i < words.length; i++) {
            const chars = [];
            for (const symbol of words[i]) {
                chars.push(symbol);
            }
            let isUnknown = false;
            let start = 0;
            const subTokens: number[] = [];
            const charsLength = chars.length;

            while (start < charsLength) {
                let end = charsLength;
                let currIndex: number | undefined = undefined;
                while (start < end) {
                    let substr = chars.slice(start, end).join('');
                    const match = this.trie!.find(substr);
                    if (match != undefined && match.end) {
                        currIndex = match.getWord()[2] as number;
                        break;
                    }
                    end = end - 1;
                }
                if (currIndex == undefined) {
                    isUnknown = true;
                    break;
                }
                subTokens.push(currIndex);
                start = end;
            }
            if (isUnknown) {
                outputTokens.push(this.UNK_INDEX);
            } else {
                outputTokens = outputTokens.concat(subTokens);
            }
        }

        return outputTokens;
    }

    public encode(text: string) {
        return [this.tokenize('[CLS]')[0], ...this.tokenize(text), this.tokenize('[SEP]')[0]]
    }
}
