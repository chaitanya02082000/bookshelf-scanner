import {mkdir, writeFile} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const environmentsDir = path.join(projectRoot, "src", "environments");

const normalizeApiUrl = (value) => {
  const trimmed = value.trim().replace(/\/+$/, "");
  return trimmed.endsWith("/api") ? trimmed : `${trimmed}/api`;
};

const devApiUrl = normalizeApiUrl(
  process.env.FRONTEND_API_URL_DEV ??
    process.env.FRONTEND_API_URL ??
    "http://localhost:8000/api"
);
const prodApiUrl = normalizeApiUrl(
  process.env.FRONTEND_API_URL_PROD ??
    process.env.FRONTEND_API_URL ??
    "http://localhost:8000/api"
);
const googleBooksApiKey =
  process.env.FRONTEND_GOOGLE_BOOKS_API_KEY ??
  "AIzaSyAiOb-JE2SRKybH6NKMW3HG_6ysmUpyf1U";

const renderEnvironment = (production, apiUrl, booksApiKey) => `export const environment = {
  production: ${production},
  apiUrl: ${JSON.stringify(apiUrl)},
  googleBooksApiKey: ${JSON.stringify(booksApiKey)},
};
`;

await mkdir(environmentsDir, {recursive: true});
await writeFile(
  path.join(environmentsDir, "environment.ts"),
  renderEnvironment(false, devApiUrl, googleBooksApiKey),
  "utf8"
);
await writeFile(
  path.join(environmentsDir, "environment.prod.ts"),
  renderEnvironment(true, prodApiUrl, googleBooksApiKey),
  "utf8"
);
