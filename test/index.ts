import {BertForMultipleChoice} from "../src";
import {BertTokenizer} from "../src/pipelines/bert/bert.tokenizer";

async function main() {
    const model = await BertForMultipleChoice.Load("ipfs://QmQUC5PdyBGZ3e6ELrgarcryK2cY5gDc5E8ao2mWeUDhk8", console.log);
    const tokenizer = await BertTokenizer.Load('ipfs://QmZrLNyDu3wXqq5s7vARUgVjWm4aUZuin13828LjAqTDgz', console.log);

    const prompt = "google.com";
    const inputIds = tokenizer.encode(prompt);

    const genClass = await model.generate(inputIds);
    console.log(genClass)
}

(async () => {
    await main();
})();
