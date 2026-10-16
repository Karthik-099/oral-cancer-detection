// Tracks scroll direction once for the whole page so every Reveal can enter
// from the side the content is coming from: below when scrolling down, above
// when scrolling back up.
let direction: 1 | -1 = 1;
let last = typeof window !== "undefined" ? window.scrollY : 0;

if (typeof window !== "undefined") {
  window.addEventListener("scroll", () => {
    const y = window.scrollY;
    if (Math.abs(y - last) > 2) direction = y > last ? 1 : -1;
    last = y;
  }, { passive: true });
}

export const scrollDirection = () => direction;
