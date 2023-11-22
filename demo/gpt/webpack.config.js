import path from "path";
import HtmlWebpackPlugin from "html-webpack-plugin";
import {fileURLToPath} from 'url';
import CopyPlugin from "copy-webpack-plugin";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default (env, argv) => ({
        target: 'web',
        context: __dirname,

        entry: {
            index: './src/index.ts',
        },

        module: {
            rules: [
                {test: /\.ts?$/, use: 'ts-loader', exclude: /node_modules/},
                {test: /\.html$/, use: 'html-loader'},
            ],
        },

        output: {
            path: __dirname + '/dist',
            filename: '[name].js',
            clean: true
        },

        resolve: {
            extensions: ['.ts', '.js', '.json'],
        },

        devServer: {
            open: ['/index.html'],
            watchFiles: ['src/*', 'node_modules/ipai/*'],
            static: {
                directory: path.join(__dirname, 'dist'),
            },
            https: true,
        },

        plugins: [
            new HtmlWebpackPlugin({
                template: './src/index.html',
                filename: 'index.html',
            }),
            new CopyPlugin({
                patterns: [{from: 'node_modules/ipai/dist/web/*.wasm', to: '[name][ext]'}]
            }),
        ],

        performance: {
            hints: false
        },

        optimization: {
            minimize: false,
        }
    }
)

