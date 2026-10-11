import { useEffect } from "react";

const SITE = "https://oral-cancer-detect.netlify.app";

/** Per-route title, description and canonical URL for crawlers and link previews. */
export function useSeo(title: string, description?: string, opts: { noindex?: boolean } = {}) {
  useEffect(() => {
    document.title = title.includes("OralAID") ? title : `${title} | OralAID`;
    const set = (selector: string, attr: string, value: string) => {
      let el = document.head.querySelector<HTMLMetaElement | HTMLLinkElement>(selector);
      if (!el) {
        const [tag, rest] = selector.split("[");
        el = document.createElement(tag) as HTMLMetaElement;
        const [k, v] = rest.replace("]", "").split("=");
        el.setAttribute(k, v.replace(/"/g, ""));
        document.head.appendChild(el);
      }
      el.setAttribute(attr, value);
    };
    if (description) {
      set('meta[name="description"]', "content", description);
      set('meta[property="og:description"]', "content", description);
    }
    set('meta[property="og:title"]', "content", document.title);
    set('link[rel="canonical"]', "href", SITE + window.location.pathname);
    set('meta[property="og:url"]', "content", SITE + window.location.pathname);
    set('meta[name="robots"]', "content", opts.noindex ? "noindex, nofollow" : "index, follow");
  }, [title, description, opts.noindex]);
}
