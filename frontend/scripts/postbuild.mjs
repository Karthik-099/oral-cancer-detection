// Runs after `vite build`. Generates files Netlify reads from the publish dir:
//   _redirects  API proxy (when ORALAID_API_ORIGIN is set) + SPA fallback
//   sitemap.xml with the public routes, stamped with the build date
// Keeping this in a script lets the backend origin come from an env var,
// which netlify.toml redirects cannot read.
import { writeFileSync } from "node:fs";

const dist = new URL("../dist/", import.meta.url);
const site = (process.env.SITE_URL || process.env.URL || "https://oralaid.ai").replace(/\/$/, "");
const api = (process.env.ORALAID_API_ORIGIN || "").replace(/\/$/, "");

const redirects = [];
if (api) {
  // 200 = rewrite (proxy). The browser only ever talks to the Netlify domain,
  // so there is no CORS and the backend origin stays private.
  redirects.push(`/api/*    ${api}/api/:splat    200!`);
  redirects.push(`/files/*  ${api}/files/:splat  200!`);
} else {
  console.warn("[postbuild] ORALAID_API_ORIGIN not set: /api will not be proxied. Set it in Netlify site settings.");
}
redirects.push("/*  /index.html  200");
writeFileSync(new URL("_redirects", dist), redirects.join("\n") + "\n");

const routes = [
  ["/", "1.0", "weekly"], ["/platform", "0.9", "monthly"], ["/models", "0.9", "monthly"],
  ["/deployment", "0.8", "monthly"], ["/pricing", "0.8", "monthly"], ["/roadmap", "0.7", "weekly"],
  ["/docs", "0.7", "monthly"], ["/security", "0.6", "monthly"], ["/about", "0.5", "yearly"],
  ["/contact", "0.5", "yearly"], ["/privacy", "0.2", "yearly"], ["/terms", "0.2", "yearly"],
];
const today = new Date().toISOString().slice(0, 10);
const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${routes.map(([p, pr, f]) => `  <url><loc>${site}${p}</loc><lastmod>${today}</lastmod><changefreq>${f}</changefreq><priority>${pr}</priority></url>`).join("\n")}
</urlset>
`;
writeFileSync(new URL("sitemap.xml", dist), sitemap);
console.log(`[postbuild] wrote _redirects (${api ? "with API proxy" : "SPA only"}) and sitemap.xml for ${site}`);
