# MemPilot project website

A responsive, static research website. Open `index.html` directly, or serve the
repository locally:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/docs/`. No build step, package manager, external
font, CDN, or API is required.

To publish with GitHub Pages, choose **Deploy from a branch**, select the branch
containing this website, and set the folder to **/docs** in the repository's Pages
settings. All assets use relative paths, so the site works
under the `/MemPilot/` project path. `.nojekyll` preserves static-file serving.

## Editing

- `index.html`: copy, authors, affiliations, resource links, and BibTeX.
- `assets/css/styles.css`: visual design and responsive layouts.
- `assets/js/data.js`: Table 1 results and Figure 3 allocation values.
- `assets/js/main.js`: benchmark/model switches, method exploration, figure
  viewer, and citation copying.
- `assets/figures/`: high-resolution paper figures and the native SVG overview
  of the orchestrator's decisions, tool delegation, and evidence feedback loop.
  The full PDF is not bundled with the website.
- `assets/logos/`: the supplied MemPilot project logo and the original NTU,
  Tsinghua, and UIUC PNG logos from the
  [MemSkill figure directory](https://github.com/ViktorAxelsen/MemSkill/tree/main/docs/static/figs),
  displayed with their original colors and proportions.

When the arXiv paper is available, replace the disabled `#paper-link` button with
a link, add the arXiv link in `.release-note`, and update the BibTeX entry and
`.citation-note`. Update the root README's arXiv badge and citation as well.
The `.paper-reference` spans retain table and section references;
they can be turned into links to the public paper at the same time.
Author order and affiliations follow the supplied author list. No acceptance
status, author homepages, or arXiv identifier is inferred from the anonymous PDF.

## Data and interpretation

- All 110 method/benchmark/model combinations in the result explorer are from
  **Table 1, page 7**, of the supplied manuscript.
- The `10/10` highlight refers specifically to the highest **LLM-Judge** score
  for MemPilot-Perf across five benchmarks and two answer models, compared with
  the methods in Table 1. It is not a claim about every metric.
- The `67%` cost reduction is MemPilot-Bal versus MemPilot-Perf on Mem-Gallery
  with Qwen3-VL-4B: `1 - 0.014 / 0.043 = 67.44%`, rounded to a whole percent.
  The accompanying Judge scores are 64.27 and 65.82.
- The result callout reports **relative** LLM-Judge improvement:
  `(MemPilot-Perf / strongest baseline - 1) * 100`, using the strongest baseline
  by Judge in the selected benchmark and answer-model setting. For the default
  view, `(65.82 / 51.45 - 1) * 100 = 27.93%`.
- Best cells are calculated independently for every view: the maximum F1,
  maximum Judge, and minimum cost. Every tied best result is highlighted.
- The shared answer model is used for fair benchmark comparisons. In normal
  use, the policy model directly generates the final answer.
- Cost is the paper's **USD/sample**, normalized by unique dataset-conversation
  pairs (Appendix A.4, page 19), not USD/question. Savings use the rounded values
  in Table 1.
- Latency is a **deterministic proxy in seconds/query**, not measured live API
  response time. The frontier image preserves the original cost-axis break.
- The allocation explorer contains **Figure 3(a,b), page 9**. Its four columns
  are separate reported policies; the controls do not imply a live runtime
  preference slider. Colors are normalized within each row.
- Ablation scores are from **Table 2, page 9**. The cost caveat is retained
  because the ablated variants also use fewer resources.
- MemEye and MEMLENS are exclusively used for out-of-distribution evaluation.
- The memory-bank compatibility card uses **Figure 4, page 9**, directly cropped
  from the manuscript, with all axes and the original legend preserved.

Keep the initial results and allocation tables in `index.html` synchronized with
`data.js` when updating the manuscript. They provide a readable no-JavaScript
fallback. Figure links remain usable without the dialog enhancement.
