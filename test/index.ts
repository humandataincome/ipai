import {BertTokenizer, BertForMultipleChoice} from "../src";

async function main() {
    const model = await BertForMultipleChoice.Load("ipfs://QmaX3qSHNe9yweJuN2YA7aMcCYCq8EZS6gHZy31X5sdz9s", console.log);
    const tokenizer = await BertTokenizer.Load('ipfs://QmZrLNyDu3wXqq5s7vARUgVjWm4aUZuin13828LjAqTDgz', console.log);

    const text = "google.com".replace('.', ' ');
    const inputIds = tokenizer.encode(text);

    const genClasses = await model.generate(inputIds);
    console.log(genClasses)
}

(async () => {
    await main();
})();
