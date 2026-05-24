import {mkdir, writeFile} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const environmentsDir = path.join(projectRoot, "src", "environments");

const devApiUrl = process.env.FRONTEND_API_URL_DEV ?? process.env.FRONTEND_API_URL ?? "http://localhost:8000/api";
const prodApiUrl = process.env.FRONTEND_API_URL_PROD ?? process.env.FRONTEND_API_URL ?? "http://localhost:8000/api";

const renderEnvironment = (production, apiUrl) => `export const environment = {
  production: ${production},
  apiUrl: ${JSON.stringify(apiUrl)},
};
`;

await mkdir(environmentsDir, {recursive: true});
await writeFile(
  path.join(environmentsDir, "environment.ts"),
  renderEnvironment(false, devApiUrl),
  "utf8"
);
await writeFile(
  path.join(environmentsDir, "environment.prod.ts"),
  renderEnvironment(true, prodApiUrl),
  "utf8"
);
