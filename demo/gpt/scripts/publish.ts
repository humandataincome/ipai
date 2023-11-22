import {create, globSource} from 'ipfs-http-client'
import * as https from "https";

async function main() {
    const username = process.env.IPFS_USERNAME;
    const password = process.env.IPFS_PASSWORD;
    const auth = "Basic " + Buffer.from(username + ":" + password).toString("base64");
    const client = create({
        host: "ipfs-api.humandataincome.com",
        port: 443,
        protocol: "https",
        headers: {
            authorization: auth,
        },
    });
    let rootCid = "";

    for await (const file of client.addAll(globSource('./dist', '**/*'), {wrapWithDirectory: true, pin: true, progress: (p) => console.log(`bytes: ${p}`)})) {
        console.log(file);
        if (file.path === "") {
            rootCid = file.cid.toString();
        }
        //await new Promise(r => setTimeout(r, 5000));
    }


    await client.files.cp("/ipfs/" + rootCid, "/" + Date.now())

    console.log("https://ipfs.humandataincome.com/ipfs/" + rootCid)




}

(async () => {
    await main();
})();

