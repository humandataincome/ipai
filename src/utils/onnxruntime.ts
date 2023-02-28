import {InferenceSession} from "onnxruntime-common"
import ort1, * as ort2 from "onnxruntime-node";

const ort = ort1 || ort2;

export type {InferenceSession}
export default ort
