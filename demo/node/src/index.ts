import {GPT2Tokenizer, GPTNeoForCausalLM} from "ipai";

async function main() {
    const model = await GPTNeoForCausalLM.Load("ipfs://QmecpDvGdWfcKw7BM4nxyEb7TB856sTY1MqY1dCR45rWjv");
    const tokenizer = await GPT2Tokenizer.Load();

    const prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.";
    const inputIds = await tokenizer.encode(prompt);

    const genTokens = await model.generate(inputIds, true, 0.9, 70);

    const genText = await tokenizer.decode(genTokens);
    console.log(genText);
}

(async () => {
    await main();
})();
