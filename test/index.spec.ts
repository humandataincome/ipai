import {GPT2Tokenizer, GPTNeoXForCausalLm} from "../src";

describe("Simple expression tests", async () => {

    it("Check 1", async () => {
        const model = await GPTNeoXForCausalLm.Load("ipfs://QmecpDvGdWfcKw7BM4nxyEb7TB856sTY1MqY1dCR45rWjv", console.log);
        const tokenizer = await GPT2Tokenizer.Load('ipfs://QmRnFHciVJxtpTtGktB3vLRMMxutEaAybXvwobXKLxRpd9', 'ipfs://QmQWBu2Cd4KnBGeeT9dx7JSG6v9VJg1QeiDg3EbBtSLKkD', console.log);

        const prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains.\nEven more surprising to the researchers was the fact that the unicorns spoke perfect English.";
        const inputIds = tokenizer.encode(prompt);

        const genTokens = await model.generate(inputIds, true, 0.9, 1, 1, 150, async t => {
            process.stdout.write(tokenizer.decode([t]))
        });

        const genText = tokenizer.decode(genTokens);
        console.log("Final text:", genText);
    }).timeout(60000);

    it("Check 2", async () => {
        const tokenizer = await GPT2Tokenizer.Load('https://huggingface.co/KoboldAI/GPT-NeoX-20B-Erebus/raw/main/vocab.json', 'https://huggingface.co/KoboldAI/GPT-NeoX-20B-Erebus/raw/main/merges.txt');

        const encoded = tokenizer.encode("Hello world,");
        console.log("encoded", encoded);

        const decoded = tokenizer.decode(encoded);
        console.log("decoded", decoded);
    }).timeout(5000);

});
