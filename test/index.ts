//import * as tflite from '@tensorflow/tfjs-tflite';

async function main() {
    //const classifier = await tflite.BertNLClassifier.create('model.tflite');
    //const probabilities = classifier.classify('github.com');
    //console.log(probabilities);

    const probabilities = [
            {probability: 0.007176657672971487, className: "-2"},
            {probability: 0.045808203518390656, className: "1"},
            {probability: 0.5899108648300171, className: "126"},
            {probability: 0.001831017667427659, className: "149"},
            {probability: 0.0007152417092584074, className: "180"}];

    const max = probabilities.reduce((p, c) => p.probability > c.probability ? p : c);
    console.log(max);

}

(async () => {
    await main();
})();
