/* Progressive enhancement for the static, build-free project website. */
(() => {
  'use strict';
  const data = window.MEMPILOT_DATA;
  const select = (selector) => document.querySelector(selector);
  const selectAll = (selector) => [...document.querySelectorAll(selector)];
  const formatCost = (value) => '$' + value.toLocaleString('en-US', { maximumFractionDigits: 5 });
  const activate = (buttons, current) => buttons.forEach((button) => {
    const active = button === current;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-pressed', String(active));
  });

  if (data) {
    let benchmarkIndex = 0;
    let modelId = 'qwen4';
    const benchmarkButtons = selectAll('[data-benchmark]');
    function renderResults() {
      const benchmark = data.benchmarks[benchmarkIndex];
      const model = data.models[modelId];
      const rows = model.rows.map((benchmarks, index) => ({ name: data.methods[index], values: benchmarks[benchmarkIndex] }));
      const baseline = rows.slice(0, 8).reduce((best, row) => row.values[1] > best.values[1] ? row : best);
      const [perf, bal, cost] = rows.slice(8);
      select('#benchmark-name').textContent = benchmark.name;
      select('#benchmark-setting').textContent = benchmark.setting;
      const series = [baseline, perf, bal, cost];
      const classes = ['baseline', 'perf', 'bal', 'cost'];
      const tags = ['Best baseline', 'Perf', 'Bal', 'Cost'];
      select('#result-bars').replaceChildren(...series.map((row, index) => {
        const element = document.createElement('div');
        element.className = `bar-row ${classes[index]}`;
        element.innerHTML = `<div class="bar-name">${index ? 'MemPilot' : row.name} <span>${tags[index]}</span></div><div class="bar-data"><div class="bar-track"><div class="bar-fill" style="width:${row.values[1]}%"></div><b${row.values[1] === Math.max(...series.map((item) => item.values[1])) ? ' class="is-best"' : ''} style="left:${row.values[1]}%">${row.values[1].toFixed(2)}</b></div><span class="bar-cost">${formatCost(row.values[2])}</span></div>`;
        return element;
      }));
      const relativeGain = (perf.values[1] / baseline.values[1] - 1) * 100;
      select('#result-delta').innerHTML = `+${relativeGain.toFixed(2)}<span>%</span>`;
      select('#result-comparison').textContent = `over ${baseline.name}`;
      select('#result-detail').innerHTML = `${perf.values[1].toFixed(2)} vs. ${baseline.values[1].toFixed(2)} LLM-Judge<br>MemPilot-Perf · ${benchmark.name}`;
      const saving = Math.round((1 - bal.values[2] / perf.values[2]) * 100);
      select('#balance-detail').innerHTML = `<strong>Bal lowers cost by ${saving}%</strong> vs. Perf, with a Judge score of ${bal.values[1].toFixed(2)} instead of ${perf.values[1].toFixed(2)}.`;
      const best = [Math.max(...rows.map((r) => r.values[0])), Math.max(...rows.map((r) => r.values[1])), Math.min(...rows.map((r) => r.values[2]))];
      select('#table-caption').textContent = `${benchmark.name} · ${model.name} · Paper, Table 1`;
      select('#results-tbody').replaceChildren(...rows.map((row, index) => {
        const tr = document.createElement('tr');
        if (index >= 8) tr.className = 'ours';
        const th = document.createElement('th');
        th.scope = 'row';
        th.textContent = row.name;
        tr.append(th);
        row.values.forEach((value, column) => {
          const td = document.createElement('td');
          td.textContent = column === 2 ? formatCost(value) : value.toFixed(2);
          if (value === best[column]) td.className = 'best';
          tr.append(td);
        });
        return tr;
      }));
    }
    benchmarkButtons.forEach((button) => button.addEventListener('click', () => {
      benchmarkIndex = Number(button.dataset.benchmark);
      activate(benchmarkButtons, button);
      renderResults();
    }));
    select('#answer-model').addEventListener('change', (event) => {
      modelId = event.target.value;
      renderResults();
    });
    renderResults();

    const allocationButtons = selectAll('[data-allocation]');
    function renderAllocation(key) {
      const setting = data.allocation[key];
      select('#allocation-caption').textContent = `${setting.unit} · five-benchmark averages`;
      select('#allocation-direction-label').textContent = key === 'cost' ? 'Lower cost' : 'Lower latency';
      select('#allocation-thead').innerHTML = `<tr><th scope="col">${setting.unit}</th>${setting.values.map((value) => `<th scope="col">${value}</th>`).join('')}</tr>`;
      select('#allocation-tbody').innerHTML = setting.rows.map((row) => {
        const max = Math.max(...row.values);
        const min = Math.min(...row.values);
        return `<tr><th scope="row">${row.name}</th>${row.values.map((value) => {
          const intensity = max === min ? .5 : (value - min) / (max - min);
          const lightness = 96 - intensity * 27;
          return `<td style="background-color:hsl(135 22% ${lightness}%)">${value.toFixed(row.digits)}${row.suffix || ''}</td>`;
        }).join('')}</tr>`;
      }).join('');
      select('#allocation-insight').textContent = key === 'cost'
        ? 'Lower-cost policies reduce curation and visual access. Evidence depth changes adaptively—it does not simply decrease at every point.'
        : 'Lower-latency policies use fewer steps and less visual curation. At the two fastest displayed settings, CURATE actions drop to zero.';
    }
    allocationButtons.forEach((button) => button.addEventListener('click', () => {
      activate(allocationButtons, button);
      renderAllocation(button.dataset.allocation);
    }));
    renderAllocation('cost');
  }

  const methods = {
    retrieve: {
      tag: 'ACTION / RETRIEVE',
      headline: 'Useful memory, without another model call.',
      description: 'Search the query-agnostic memory bank with a learned retrieval query and evidence count. The bank can come from an existing memory system; retrieved text returns directly to the orchestrator.',
      controls: [['Retrieval query', 'r'], ['Evidence count', 'k']],
      note: 'The default bank uses LLMLingua-2. Raw text and images remain available for later curation.'
    },
    curate: {
      tag: 'ACTION / CURATE',
      headline: 'Go back to the details that matter.',
      description: 'Retrieve relevant chunks of raw history, then delegate query-specific processing to an LLM or VLM. Separately choose what to retrieve, how to curate it, which model to use, and whether to include original images.',
      controls: [['Retrieval query', 'r'], ['Evidence count', 'k'], ['Curation instruction', 'i'], ['Model selection', 'm'], ['Visual access', 'v']],
      note: 'The orchestrator is text-only. A selected vision-capable model handles images and returns textual evidence.'
    },
    answer: {
      tag: 'DECISION / CONTINUE OR ANSWER',
      headline: 'Enough evidence? Bring it together.',
      description: 'After each memory operation, the policy can retrieve again, delegate further curation, or directly generate the final answer from the accumulated evidence.',
      controls: [['Query', 'q'], ['Accumulated evidence', 'o'], ['Final answer', 'y']],
      note: 'No separate answer model is needed in normal use.'
    }
  };
  const methodButtons = selectAll('[data-step]');
  methodButtons.forEach((button) => button.addEventListener('click', () => {
    activate(methodButtons, button);
    const method = methods[button.dataset.step];
    select('#method-tag').textContent = method.tag;
    select('#method-headline').textContent = method.headline;
    select('#method-description').textContent = method.description;
    select('#method-controls').innerHTML = method.controls.map(([label, symbol]) => `<span>${label} <i>${symbol}</i></span>`).join('');
    select('#method-footnote').textContent = method.note;
  }));

  const dialog = select('#figure-dialog');
  let previousFocus = null;
  if (typeof dialog.showModal === 'function') {
    selectAll('.figure-open').forEach((link) => link.addEventListener('click', (event) => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      previousFocus = link;
      select('#dialog-image').src = link.href;
      select('#dialog-image').alt = link.querySelector('img').alt;
      select('#dialog-title').textContent = link.dataset.figureTitle;
      select('#dialog-original').href = link.href;
      dialog.showModal();
      document.body.classList.add('has-dialog');
      select('#close-dialog').focus();
    }));
    select('#close-dialog').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', (event) => {
      const rect = dialog.getBoundingClientRect();
      if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
    });
    dialog.addEventListener('close', () => {
      document.body.classList.remove('has-dialog');
      if (previousFocus) previousFocus.focus({ preventScroll: true });
    });
  }

  const copyButton = select('#copy-citation');
  let copyTimer;
  copyButton.addEventListener('click', async () => {
    const text = select('#bibtex').textContent;
    let copied = false;
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
    } catch {
      // Preserve file:// support and provide a manual selection if copying is denied.
      const field = document.createElement('textarea');
      field.value = text;
      field.setAttribute('aria-label', 'Citation to copy');
      field.style.cssText = 'position:fixed;left:-9999px;top:0;';
      document.body.append(field);
      field.select();
      try { copied = document.execCommand('copy'); } catch { copied = false; }
      field.remove();
      copyButton.focus({ preventScroll: true });
    }
    clearTimeout(copyTimer);
    if (copied) {
      copyButton.querySelector('span').textContent = 'Copied!';
      select('#copy-status').textContent = 'BibTeX copied to clipboard.';
    } else {
      const range = document.createRange();
      range.selectNodeContents(select('#bibtex'));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copyButton.querySelector('span').textContent = 'Select & copy';
      select('#copy-status').textContent = 'Citation selected. Press Control+C or Command+C to copy.';
    }
    copyTimer = setTimeout(() => {
      copyButton.querySelector('span').textContent = 'Copy';
      select('#copy-status').textContent = '';
    }, 3500);
  });

  // Keep navigation orientation without hiding content or depending on animation.
  if ('IntersectionObserver' in window) {
    const navLinks = selectAll('.site-header nav a');
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        navLinks.forEach((link) => {
          const current = link.hash === '#' + entry.target.id;
          link.classList.toggle('is-current', current);
          if (current) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      });
    }, { rootMargin: '-15% 0px -60% 0px', threshold: 0 });
    observer.observe(select('.hero'));
    ['idea', 'method', 'results', 'frontiers', 'analysis', 'citation'].forEach((id) => observer.observe(document.getElementById(id)));
  }
})();
