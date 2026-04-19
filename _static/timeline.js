// Fade-up the timeline cards as they scroll into view.
// Completely skipped when the reader has `prefers-reduced-motion: reduce`.
(() => {
  "use strict";

  const init = () => {
    const entries = document.querySelectorAll(".kb-entry");
    if (entries.length === 0) return;

    const reduceMotion =
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (reduceMotion || typeof IntersectionObserver === "undefined") {
      entries.forEach((el) => el.classList.add("kb-in-view"));
      return;
    }

    const obs = new IntersectionObserver(
      (records, self) => {
        records.forEach((rec) => {
          if (rec.isIntersecting) {
            rec.target.classList.add("kb-in-view");
            self.unobserve(rec.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: "0px 0px -40px 0px" },
    );

    entries.forEach((el) => obs.observe(el));
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
