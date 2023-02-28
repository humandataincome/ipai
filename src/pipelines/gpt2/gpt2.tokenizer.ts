// References
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py
// https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py

import * as ipdw from "ipdw";

const range = (x: number, y: number): number[] => Array.from(Array(y).keys()).slice(x);
const ord = (x: string): number => x.charCodeAt(0);
const chr = (x: number): string => String.fromCharCode(x);
const min = <T>(a: T[], k: (v: T) => number): T | undefined => a.length > 0 ? a.reduce((p, c) => k(c) < k(p) ? c : p) : undefined;

function bytesToUnicode(): { [k: number]: string } {
    const bs = range(ord('!'), ord('~') + 1).concat(range(ord('¡'), ord('¬') + 1), range(ord('®'), ord('ÿ') + 1));

    let cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 2 ** 8; b++) {
        if (!bs.includes(b)) {
            bs.push(b);
            cs.push(2 ** 8 + n);
            n = n + 1;
        }
    }

    const cs1 = cs.map(x => chr(x));
    return Object.fromEntries(bs.map((v, i) => [v, cs1[i]]));
}

function getPairs(word: string[]): Set<string[]> {
    const pairs = new Set<string[]>();
    let prevChar = word[0];
    for (const char of word.slice(1)) {
        pairs.add([prevChar, char]);
        prevChar = char;
    }
    return pairs;
}

export class GPT2Tokenizer {
    private readonly encoder: { [k: string]: number };
    private readonly decoder: { [k: number]: string };
    private readonly byteEncoder: { [k: number]: string };
    private readonly byteDecoder: { [k: string]: number };
    private readonly bpeRanks: { [k: string]: number };
    private readonly cache: { [k: string]: string };
    private readonly pat: RegExp;

    constructor(vocab: string, merges: string) {
        this.encoder = JSON.parse(vocab);
        this.decoder = Object.fromEntries(Object.entries(this.encoder).map(([k, v]) => [v, k]));
        this.byteEncoder = bytesToUnicode();
        this.byteDecoder = Object.fromEntries(Object.entries(this.byteEncoder).map(([k, v]) => [v, Number(k)]));
        const bpeMerges = merges.split('\n').slice(1, -1);
        const bpeMerges1 = bpeMerges.map(x => x.split(/(\s+)/).filter((e) => e.trim().length > 0));
        this.bpeRanks = Object.fromEntries(Object.entries(bpeMerges1).map(([k, v], i) => [v, i]));
        this.cache = {};
        this.pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    }

    public static async Load(vocabPath: string, mergesPath: string, progress?: ((progress: number) => void) | undefined): Promise<GPT2Tokenizer> {
        const persistence = await ipdw.Persistence.getInstance();
        const vocab = await persistence.fetchOrGet(vocabPath, progress ? p => progress(p * (1 / 2)) : undefined);
        const merges = await persistence.fetchOrGet(mergesPath, progress ? p => progress(p * (1 / 2) + (1 / 2)) : undefined);

        return new GPT2Tokenizer(new TextDecoder().decode(vocab), new TextDecoder().decode(merges));
    }

    public bpe(token: string): string {
        if (token in this.cache)
            return this.cache[token];

        let word = token.split('');
        let pairs = getPairs(word);

        if (!pairs)
            return token;

        while (true) {
            const bigram = min(Array.from(pairs), v => this.bpeRanks[v as any] ?? Number.MAX_VALUE);

            if (!(bigram as any in this.bpeRanks))
                break;

            const [first, second] = bigram!;
            let newWord: string[] = [];
            let i = 0;
            while (i < word.length) {
                const j = word.indexOf(first, i);
                if (j === -1) {
                    newWord = newWord.concat(word.slice(i));
                    break;
                }
                newWord = newWord.concat(word.slice(i, j));
                i = j;

                if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
                    newWord.push(first + second);
                    i += 2;
                } else {
                    newWord.push(word[i]);
                    i += 1;
                }
            }

            word = newWord;
            if (word.length === 1)
                break;
            else
                pairs = getPairs(word);
        }

        const word1 = word.join(' ');
        this.cache[token] = word1;

        return word1;
    }

    public encode(text: string): number[] {
        let bpe_tokens: number[] = []
        //text = text.toLowerCase();
        const textEncoder = new TextEncoder();
        for (let token of Array.from(text.matchAll(this.pat)).map(x => x[0])) {
            token = Array.from(textEncoder.encode(token)).map(b => this.byteEncoder[b]).join('');
            bpe_tokens = bpe_tokens.concat(this.bpe(token).split(' ').map(x => this.encoder[x]));
        }
        return bpe_tokens;
    }

    public decode(tokens: number[]): string {
        let text = tokens.map(x => this.decoder[x]).join('');
        const textDecoder = new TextDecoder();
        text = textDecoder.decode(new Uint8Array(text.split('').map(x => this.byteDecoder[x])));
        return text
    }
}
