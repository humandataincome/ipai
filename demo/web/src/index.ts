import {GPT2Tokenizer, GPTNeoXForCausalLm} from "ipai";

async function main() {
    let inputContent = document.getElementById('input-content') as HTMLDivElement;
    let runButton = document.getElementById('run-button') as HTMLButtonElement;

    let modelParam = document.getElementById('model-param') as HTMLDivElement;
    let temperatureParam = document.getElementById('temperature-param') as HTMLInputElement;
    let maxlengthParam = document.getElementById('maxlength-param') as HTMLInputElement;
    let sequencesParam = document.getElementById('sequences-param') as HTMLInputElement;
    let toppParam = document.getElementById('topp-param') as HTMLInputElement;
    let freqpenaltyParam = document.getElementById('freqpenalty-param') as HTMLInputElement;
    let prespenaltyParam = document.getElementById('prespenalty-param') as HTMLInputElement;
    let topkParam = document.getElementById('topk-param') as HTMLInputElement;

    runButton.disabled = true;
    inputContent.setAttribute("data-text", "Loading...");
    //inputContent.contentEditable = "false";
    modelParam.textContent = "Loading...";

    const model = await GPTNeoXForCausalLm.Load("ipfs://QmecpDvGdWfcKw7BM4nxyEb7TB856sTY1MqY1dCR45rWjv", p => setTimeout(() => inputContent.setAttribute("data-text", `Model loading... ${p}%}`), 0));
    const tokenizer = await GPT2Tokenizer.Load('ipfs://QmRnFHciVJxtpTtGktB3vLRMMxutEaAybXvwobXKLxRpd9', 'ipfs://QmQWBu2Cd4KnBGeeT9dx7JSG6v9VJg1QeiDg3EbBtSLKkD', p => inputContent.setAttribute("data-text", `Tokenizer loading... ${p}%}`));

    runButton.disabled = false;
    inputContent.setAttribute("data-text", "Model loaded. Please ask me something");
    //inputContent.contentEditable = "true";
    modelParam.textContent = "ipfs://QmecpDvGdWfcKw7BM4nxyEb7TB856sTY1MqY1dCR45rWjv";

    runButton!.onclick = async () => {
        try {
            const inputIds = tokenizer.encode(inputContent.textContent!);
            runButton.disabled = true;
            await model.generate(
                inputIds,
                true,
                +temperatureParam.value,
                +topkParam.value,
                +toppParam.value,
                +maxlengthParam.value,
                async t => {
                    inputContent.textContent += tokenizer.decode([t]);
                    await waitForNextAnimationFrame();
                });
        } catch (_) {
        }
        runButton.disabled = false;
    }
}

function waitForNextAnimationFrame(): Promise<void> {
    return new Promise((resolve) => {
        window.requestAnimationFrame(() => resolve());
    });
}


(async () => {
    await main();
})();
