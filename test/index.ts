import {BertTokenizer, BertForMultipleChoice} from "../src";

async function main() {
    const model = await BertForMultipleChoice.Load("https://asset.humandataincome.com/ipai/bdc1/bert_domain_classifier.onnx", console.log);
    const tokenizer = await BertTokenizer.Load('https://asset.humandataincome.com/ipai/bdc1/vocab.json', console.log);
    //const model = await BertForMultipleChoice.Load("ipfs://QmfByoob3ZNgXpC14bxRjGAPEKLSYJRKE5uhhNa6T68JQf", console.log);
    //const tokenizer = await BertTokenizer.Load('ipfs://QmZrLNyDu3wXqq5s7vARUgVjWm4aUZuin13828LjAqTDgz', console.log);

    const text = "google.com".replace('.', ' ');
    const inputIds = tokenizer.encode(text);

    const genClasses = await model.generate(inputIds);
    console.log(genClasses)
}

(async () => {
    await main();
})();
