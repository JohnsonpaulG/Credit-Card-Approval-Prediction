// ============================================================
// CreditAI — shared front-end interactions
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  initNavToggle();
  initScrollReveal();
  initCounters();
  initModelTabs();
  initImportanceBars();
  initHistoryTable();
});

// ---------------------------------------------------------------
// Mobile nav toggle
// ---------------------------------------------------------------
function initNavToggle() {
  const toggle = document.querySelector('.nav-toggle');
  const links = document.querySelector('.nav-links');
  if (!toggle || !links) return;
  toggle.addEventListener('click', () => {
    links.classList.toggle('open');
    toggle.textContent = links.classList.contains('open') ? '✕' : '☰';
  });
}

// ---------------------------------------------------------------
// Scroll reveal for elements marked with .reveal
// ---------------------------------------------------------------
function initScrollReveal() {
  const items = document.querySelectorAll('.reveal');
  if (!items.length) return;

  if (!('IntersectionObserver' in window)) {
    items.forEach(el => el.classList.add('is-visible'));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });

  items.forEach(el => observer.observe(el));
}

// ---------------------------------------------------------------
// Animated counters — any element with [data-counter] gets animated
// from 0 to the value in data-counter, respecting data-prefix/suffix
// and data-decimals.
// ---------------------------------------------------------------
function animateCounter(el) {
  const target = parseFloat(el.dataset.counter);
  if (Number.isNaN(target)) return;
  const decimals = parseInt(el.dataset.decimals || '0', 10);
  const prefix = el.dataset.prefix || '';
  const suffix = el.dataset.suffix || '';
  const duration = 1400;
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = target * eased;
    el.textContent = prefix + value.toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(tick);
    else el.textContent = prefix + target.toFixed(decimals) + suffix;
  }
  requestAnimationFrame(tick);
}

function initCounters() {
  const counters = document.querySelectorAll('[data-counter]');
  if (!counters.length) return;

  if (!('IntersectionObserver' in window)) {
    counters.forEach(animateCounter);
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.4 });

  counters.forEach(el => observer.observe(el));
}

// ---------------------------------------------------------------
// ML Models page — tab switching
// ---------------------------------------------------------------
function initModelTabs() {
  const tabs = document.querySelectorAll('.model-tab');
  if (!tabs.length) return;

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.target;

      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      document.querySelectorAll('.model-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === target);
      });

      // Re-trigger bar animations within the newly shown panel
      const panel = document.getElementById(target);
      if (panel) {
        panel.querySelectorAll('.importance-bar-fill').forEach(bar => {
          bar.style.width = '0';
          requestAnimationFrame(() => {
            bar.style.width = bar.dataset.width;
          });
        });
      }
    });
  });
}

// ---------------------------------------------------------------
// Feature importance bar animation on first view
// ---------------------------------------------------------------
function initImportanceBars() {
  const bars = document.querySelectorAll('.importance-bar-fill');
  if (!bars.length) return;

  if (!('IntersectionObserver' in window)) {
    bars.forEach(bar => { bar.style.width = bar.dataset.width; });
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.width = entry.target.dataset.width;
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });

  bars.forEach(bar => observer.observe(bar));
}

// ---------------------------------------------------------------
// History page — client-side search, filter, sort, pagination
// ---------------------------------------------------------------
function initHistoryTable() {
  const table = document.getElementById('historyTable');
  if (!table) return;

  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const searchInput = document.getElementById('historySearch');
  const decisionFilter = document.getElementById('historyDecisionFilter');
  const riskFilter = document.getElementById('historyRiskFilter');
  const pageSize = 10;
  let currentPage = 1;
  let sortKey = null;
  let sortAsc = true;

  function getFiltered() {
    const query = (searchInput?.value || '').toLowerCase().trim();
    const decision = decisionFilter?.value || '';
    const risk = riskFilter?.value || '';

    return rows.filter(row => {
      const matchesQuery = !query || row.dataset.search.includes(query);
      const matchesDecision = !decision || row.dataset.decision === decision;
      const matchesRisk = !risk || row.dataset.risk === risk;
      return matchesQuery && matchesDecision && matchesRisk;
    });
  }

  function applySort(list) {
    if (!sortKey) return list;
    return [...list].sort((a, b) => {
      const av = a.dataset[sortKey];
      const bv = b.dataset[sortKey];
      const an = parseFloat(av);
      const bn = parseFloat(bv);
      let cmp;
      if (!Number.isNaN(an) && !Number.isNaN(bn)) {
        cmp = an - bn;
      } else {
        cmp = av.localeCompare(bv);
      }
      return sortAsc ? cmp : -cmp;
    });
  }

  function render() {
    rows.forEach(row => { row.style.display = 'none'; });

    let filtered = applySort(getFiltered());
    const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
    currentPage = Math.min(currentPage, totalPages);

    const startIdx = (currentPage - 1) * pageSize;
    const pageRows = filtered.slice(startIdx, startIdx + pageSize);

    pageRows.forEach(row => { row.style.display = ''; });

    renderPagination(totalPages, filtered.length);
  }

  function renderPagination(totalPages, totalCount) {
    const container = document.getElementById('historyPagination');
    if (!container) return;
    container.innerHTML = '';

    const summary = document.createElement('span');
    summary.textContent = `${totalCount} result${totalCount === 1 ? '' : 's'}`;
    container.appendChild(summary);

    const prev = document.createElement('button');
    prev.textContent = '← Prev';
    prev.disabled = currentPage === 1;
    prev.addEventListener('click', () => { currentPage -= 1; render(); });
    container.appendChild(prev);

    for (let p = 1; p <= totalPages; p += 1) {
      const btn = document.createElement('button');
      btn.textContent = String(p);
      if (p === currentPage) btn.classList.add('active');
      btn.addEventListener('click', () => { currentPage = p; render(); });
      container.appendChild(btn);
    }

    const next = document.createElement('button');
    next.textContent = 'Next →';
    next.disabled = currentPage === totalPages;
    next.addEventListener('click', () => { currentPage += 1; render(); });
    container.appendChild(next);
  }

  searchInput?.addEventListener('input', () => { currentPage = 1; render(); });
  decisionFilter?.addEventListener('change', () => { currentPage = 1; render(); });
  riskFilter?.addEventListener('change', () => { currentPage = 1; render(); });

  table.querySelectorAll('th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      if (sortKey === key) {
        sortAsc = !sortAsc;
      } else {
        sortKey = key;
        sortAsc = true;
      }
      render();
    });
  });

  render();
}
