// IMAP-LLM Web UI - JavaScript

// API Configuration
const API_BASE = '';  // Same origin

// Fetch options to include HTTP Basic Auth credentials
const FETCH_OPTIONS = {
    credentials: 'include'  // Include cookies and HTTP Basic Auth
};

// State
let currentTab = 'search';
let processingCheckInterval = null;
let accounts = [];
let selectedAccount = '';

// Search history configuration
const SEARCH_HISTORY_KEY = 'mail-done-search-history';
const MAX_SEARCH_HISTORY = 15;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async function() {
    // Check authentication FIRST before initializing anything
    const isAuthenticated = await checkAuthentication();
    
    if (!isAuthenticated) {
        // Show login screen, hide app - already handled by checkAuthentication
        return;
    }
    
    // User is authenticated - initialize the app
    initTabs();
    initSearch();
    initBrowse();
    initCosts();
    initProcess();
    initStats();
    initAccounts();
    initUserInfo();
    checkConnection();
});

// Check if user is authenticated and show/hide screens accordingly
async function checkAuthentication() {
    const loginScreen = document.getElementById('login-screen');
    const appContainer = document.getElementById('app-container');
    
    try {
        const response = await fetch(`${API_BASE}/auth/status`, FETCH_OPTIONS);
        if (response.ok) {
            const data = await response.json();
            
            if (data.authenticated) {
                // Hide login, show app
                loginScreen.classList.add('hidden');
                appContainer.classList.add('visible');
                document.title = 'Mail-Done Web UI';
                return true;
            }
        }
    } catch (error) {
        console.log('Auth check failed:', error);
    }
    
    // Not authenticated - show login screen (already visible by default)
    loginScreen.classList.remove('hidden');
    appContainer.classList.remove('visible');
    document.title = 'Login';
    return false;
}

// ============================================================================
// User Info & Logout
// ============================================================================

async function initUserInfo() {
    try {
        const response = await fetch(`${API_BASE}/auth/status`, FETCH_OPTIONS);
        if (response.ok) {
            const data = await response.json();
            const userEmailEl = document.getElementById('user-email');
            const logoutButton = document.getElementById('logout-button');
            
            if (data.authenticated && data.user_email) {
                userEmailEl.textContent = data.user_email;
                
                // Add logout handler
                logoutButton.addEventListener('click', handleLogout);
            }
        }
    } catch (error) {
        console.log('Error fetching user info:', error);
    }
}

async function handleLogout() {
    const logoutButton = document.getElementById('logout-button');
    logoutButton.disabled = true;
    logoutButton.textContent = '⏳ Logging out...';
    
    try {
        const response = await fetch(`${API_BASE}/auth/logout`, {
            method: 'GET',
            credentials: 'include'
        });
        
        if (response.ok || response.redirected) {
            // Redirect to login page
            window.location.href = '/auth/login';
        } else {
            // Force redirect anyway
            window.location.href = '/auth/login';
        }
    } catch (error) {
        console.error('Logout error:', error);
        // Force redirect anyway
        window.location.href = '/auth/login';
    }
}

// ============================================================================
// Account Management
// ============================================================================

async function initAccounts() {
    try {
        const response = await fetch(`${API_BASE}/api/accounts`, FETCH_OPTIONS);
        if (response.ok) {
            const data = await response.json();
            accounts = data.accounts || [];
            
            const selector = document.getElementById('account-selector');
            selector.innerHTML = '<option value="">All Accounts</option>';
            
            accounts.forEach(account => {
                const option = document.createElement('option');
                option.value = account.nickname;
                option.textContent = `${account.display_name}${account.is_default ? ' (default)' : ''}`;
                selector.appendChild(option);
            });
            
            // Add change handler
            selector.addEventListener('change', function() {
                selectedAccount = this.value;
                // Reload current tab data
                if (currentTab === 'browse') {
                    loadCategories();
                } else if (currentTab === 'search') {
                    // Trigger search if there's a query
                    const query = document.getElementById('search-query').value;
                    if (query) {
                        performSearch();
                    }
                }
            });
        }
    } catch (error) {
        console.error('Failed to load accounts:', error);
    }
}

// ============================================================================
// Connection Check
// ============================================================================

async function checkConnection() {
    const statusEl = document.getElementById('connection-status');
    
    try {
        const response = await fetch(`${API_BASE}/health`, FETCH_OPTIONS);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusEl.textContent = '✅ Connected to Backend API';
            statusEl.classList.add('connected');
            statusEl.classList.remove('disconnected');
        } else {
            statusEl.textContent = '⚠️ Backend API degraded';
            statusEl.classList.add('disconnected');
            statusEl.classList.remove('connected');
        }
    } catch (error) {
        statusEl.textContent = '❌ Cannot connect to Backend API';
        statusEl.classList.add('disconnected');
        statusEl.classList.remove('connected');
    }
}

// ============================================================================
// Tab Navigation
// ============================================================================

function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // Update panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    currentTab = tabName;
    
    // Load data for tab
    if (tabName === 'browse') {
        loadCategories();
    } else if (tabName === 'costs') {
        loadCosts();
    } else if (tabName === 'stats') {
        // Initialize stats sub-tabs when switching to stats tab
        initStatsSubtabs();
        loadStats();
    } else if (tabName === 'process') {
        checkProcessingStatus();
    }
}

// ============================================================================
// Search Tab
// ============================================================================

function initSearch() {
    const searchButton = document.getElementById('search-button');
    const searchQuery = document.getElementById('search-query');
    const dateRangeSelect = document.getElementById('search-date-range');
    const customDateDiv = document.getElementById('custom-date-range');
    
    // Toggle custom date range inputs
    dateRangeSelect.addEventListener('change', (e) => {
        if (e.target.value === 'custom') {
            customDateDiv.style.display = 'flex';
        } else {
            customDateDiv.style.display = 'none';
        }
    });
    
    searchButton.addEventListener('click', performSearch);
    
    // Allow Enter key to search
    searchQuery.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    
    // Initialize search history dropdown
    initSearchHistory();
}

// ============================================================================
// Search History
// ============================================================================

function initSearchHistory() {
    const searchQuery = document.getElementById('search-query');
    const searchForm = searchQuery.closest('.form-group');
    
    // Create history dropdown container
    const historyDropdown = document.createElement('div');
    historyDropdown.id = 'search-history-dropdown';
    historyDropdown.className = 'search-history-dropdown';
    historyDropdown.style.display = 'none';
    searchForm.style.position = 'relative';
    searchForm.appendChild(historyDropdown);
    
    // Show history on focus
    searchQuery.addEventListener('focus', () => {
        showSearchHistory();
    });
    
    // Hide history on blur (with delay for click)
    searchQuery.addEventListener('blur', () => {
        setTimeout(() => {
            hideSearchHistory();
        }, 200);
    });
    
    // Filter history as user types
    searchQuery.addEventListener('input', () => {
        showSearchHistory();
    });
    
    // Keyboard navigation
    searchQuery.addEventListener('keydown', (e) => {
        const dropdown = document.getElementById('search-history-dropdown');
        if (dropdown.style.display === 'none') return;
        
        const items = dropdown.querySelectorAll('.search-history-item');
        const activeItem = dropdown.querySelector('.search-history-item.active');
        let activeIndex = Array.from(items).indexOf(activeItem);
        
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (activeIndex < items.length - 1) {
                if (activeItem) activeItem.classList.remove('active');
                items[activeIndex + 1].classList.add('active');
            } else if (activeIndex === -1 && items.length > 0) {
                items[0].classList.add('active');
            }
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (activeIndex > 0) {
                if (activeItem) activeItem.classList.remove('active');
                items[activeIndex - 1].classList.add('active');
            }
        } else if (e.key === 'Enter' && activeItem) {
            e.preventDefault();
            const query = activeItem.dataset.query;
            searchQuery.value = query;
            hideSearchHistory();
            performSearch();
        } else if (e.key === 'Escape') {
            hideSearchHistory();
        }
    });
}

function getSearchHistory() {
    try {
        const history = localStorage.getItem(SEARCH_HISTORY_KEY);
        return history ? JSON.parse(history) : [];
    } catch (e) {
        console.error('Failed to load search history:', e);
        return [];
    }
}

function saveSearchToHistory(query) {
    if (!query || query.trim().length === 0) return;
    
    query = query.trim();
    
    try {
        let history = getSearchHistory();
        
        // Remove duplicate if exists
        history = history.filter(item => item.query.toLowerCase() !== query.toLowerCase());
        
        // Add new query at the beginning with timestamp
        history.unshift({
            query: query,
            timestamp: Date.now()
        });
        
        // Keep only MAX_SEARCH_HISTORY items
        if (history.length > MAX_SEARCH_HISTORY) {
            history = history.slice(0, MAX_SEARCH_HISTORY);
        }
        
        localStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(history));
    } catch (e) {
        console.error('Failed to save search history:', e);
    }
}

function deleteSearchHistoryItem(query) {
    try {
        let history = getSearchHistory();
        history = history.filter(item => item.query !== query);
        localStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(history));
        showSearchHistory(); // Refresh dropdown
    } catch (e) {
        console.error('Failed to delete search history item:', e);
    }
}

function clearSearchHistory() {
    try {
        localStorage.removeItem(SEARCH_HISTORY_KEY);
        hideSearchHistory();
    } catch (e) {
        console.error('Failed to clear search history:', e);
    }
}

function showSearchHistory() {
    const dropdown = document.getElementById('search-history-dropdown');
    const searchQuery = document.getElementById('search-query');
    const currentValue = searchQuery.value.toLowerCase().trim();
    
    let history = getSearchHistory();
    
    // Filter by current input if any
    if (currentValue) {
        history = history.filter(item => 
            item.query.toLowerCase().includes(currentValue)
        );
    }
    
    if (history.length === 0) {
        dropdown.style.display = 'none';
        return;
    }
    
    // Build dropdown HTML
    let html = '<div class="search-history-header">';
    html += '<span>🕒 Recent Searches</span>';
    html += '<button class="search-history-clear" onclick="event.stopPropagation(); clearSearchHistory();" title="Clear all">Clear</button>';
    html += '</div>';
    
    html += history.map(item => {
        const escapedQuery = escapeHtml(item.query);
        const timeAgo = formatTimeAgo(item.timestamp);
        
        return `
            <div class="search-history-item" data-query="${escapedQuery}" onclick="selectSearchHistoryItem('${escapedQuery.replace(/'/g, "\\'")}')">
                <span class="search-history-query">🔍 ${escapedQuery}</span>
                <span class="search-history-meta">
                    <span class="search-history-time">${timeAgo}</span>
                    <button class="search-history-delete" onclick="event.stopPropagation(); deleteSearchHistoryItem('${escapedQuery.replace(/'/g, "\\'")}');" title="Remove">×</button>
                </span>
            </div>
        `;
    }).join('');
    
    dropdown.innerHTML = html;
    dropdown.style.display = 'block';
}

function hideSearchHistory() {
    const dropdown = document.getElementById('search-history-dropdown');
    if (dropdown) {
        dropdown.style.display = 'none';
    }
}

function selectSearchHistoryItem(query) {
    const searchQuery = document.getElementById('search-query');
    searchQuery.value = query;
    hideSearchHistory();
    performSearch();
}

function formatTimeAgo(timestamp) {
    const now = Date.now();
    const diff = now - timestamp;
    
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return new Date(timestamp).toLocaleDateString();
}

async function performSearch() {
    const query = document.getElementById('search-query').value.trim();
    const searchType = document.getElementById('search-type').value;
    const mode = document.getElementById('search-mode').value;
    const limit = parseInt(document.getElementById('search-limit').value);
    const thresholdInput = document.getElementById('search-threshold').value;
    const threshold = thresholdInput ? parseFloat(thresholdInput) : 0.0;
    const hideHandled = document.getElementById('search-hide-handled')?.checked ?? false;
    const dateRange = document.getElementById('search-date-range').value;
    const resultsEl = document.getElementById('search-results');
    const searchButton = document.getElementById('search-button');

    if (!query) {
        showError(resultsEl, 'Please enter a search query');
        return;
    }

    // Hide search history dropdown
    hideSearchHistory();

    // Save to search history
    saveSearchToHistory(query);

    // Show loading
    searchButton.disabled = true;
    searchButton.textContent = 'Searching...';
    resultsEl.innerHTML = '<div class="loading">Searching</div>';

    try {
        // Build URL with proper threshold (use 0.01 instead of 0 to avoid filtering all results)
        const effectiveThreshold = threshold === 0 ? 0.01 : threshold;

        // Use unified search for document or all types, simple search for email-only
        let url;
        let isUnifiedSearch = (searchType === 'document' || searchType === 'all');

        if (isUnifiedSearch) {
            // Use unified search endpoint
            url = `${API_BASE}/api/search/unified?q=${encodeURIComponent(query)}&types=${searchType}&limit=${limit}&similarity_threshold=${effectiveThreshold}`;

            // Add account filter as email_account for unified search
            if (selectedAccount) {
                url += `&email_account=${encodeURIComponent(selectedAccount)}`;
            }
        } else {
            // Use simple search for emails only (supports hybrid/keyword modes)
            url = `${API_BASE}/api/search/simple?q=${encodeURIComponent(query)}&mode=${mode}&limit=${limit}&similarity_threshold=${effectiveThreshold}`;

            // Add account filter
            if (selectedAccount) {
                url += `&account=${encodeURIComponent(selectedAccount)}`;
            }

            // Add exclude_handled if checkbox is checked
            if (hideHandled) {
                url += '&exclude_handled=true';
            }
        }

        // Add date range filters (both endpoints support these)
        if (dateRange !== 'all') {
            if (dateRange === 'custom') {
                const dateFrom = document.getElementById('search-date-from').value;
                const dateTo = document.getElementById('search-date-to').value;
                if (dateFrom) {
                    url += `&date_from=${encodeURIComponent(dateFrom)}`;
                }
                if (dateTo) {
                    url += `&date_to=${encodeURIComponent(dateTo)}`;
                }
            } else {
                // Calculate date from days ago
                const daysAgo = parseInt(dateRange);
                const dateFrom = new Date();
                dateFrom.setDate(dateFrom.getDate() - daysAgo);
                url += `&date_from=${encodeURIComponent(dateFrom.toISOString().split('T')[0])}`;
            }
        }

        console.log('Search URL:', url);
        const response = await fetch(url, FETCH_OPTIONS);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Search failed (${response.status}): ${errorText}`);
        }

        const data = await response.json();
        console.log('Search results:', data);

        // Use appropriate display function based on search type
        if (isUnifiedSearch) {
            displayUnifiedSearchResults(data, resultsEl);
        } else {
            displaySearchResults(data, resultsEl);
        }

    } catch (error) {
        console.error('Search error:', error);
        showError(resultsEl, `Search failed: ${error.message}`);
    } finally {
        searchButton.disabled = false;
        searchButton.textContent = 'Search';
    }
}

function displaySearchResults(data, container) {
    console.log('Display results - raw data:', data);
    
    if (!data.results || data.results.length === 0) {
        container.innerHTML = `
            <div class="error-message">
                <strong>No results found.</strong><br>
                <br>
                Try adjusting your search:<br>
                • Lower the similarity threshold (try 0.1)<br>
                • Use a different search mode<br>
                • Try broader search terms<br>
            </div>
        `;
        return;
    }
    
    // Sort results by score descending (highest first)
    const sortedResults = [...data.results].sort((a, b) => (b.score || 0) - (a.score || 0));
    
    const resultsHtml = sortedResults.map((result, index) => {
        // Backend API returns nested structure: { email: {...}, score: X }
        const email = result.email || result;
        const score = result.score || 0;
        
        // Highlight top 3 results
        const isTopResult = index < 3;
        const topBadge = isTopResult ? `<span class="top-badge">Top ${index + 1}</span>` : '';
        
        // Don't show "Wrong Category" button in search (searching across categories)
        return renderEmailItem(email, index, `search-${index}`, score, topBadge, false);
    }).join('');
    
    // Show stats about results
    const avgScore = sortedResults.reduce((sum, r) => sum + (r.score || 0), 0) / sortedResults.length;
    const topScore = sortedResults[0]?.score || 0;
    
    container.innerHTML = `
        <div class="success-message">
            Found ${data.total || data.results.length} results for "${escapeHtml(data.query)}" using ${data.mode} mode
            <br>
            <small>Top score: ${(topScore * 100).toFixed(1)}% | Average: ${(avgScore * 100).toFixed(1)}%</small>
        </div>
        ${resultsHtml}
    `;
}

// Display unified search results (emails + files)
function displayUnifiedSearchResults(data, container) {
    console.log('Display unified results - raw data:', data);

    if (!data.results || data.results.length === 0) {
        container.innerHTML = `
            <div class="error-message">
                <strong>No results found.</strong><br>
                <br>
                Try adjusting your search:<br>
                • Lower the similarity threshold (try 0.3)<br>
                • Search in "All" instead of just emails or documents<br>
                • Try broader search terms<br>
            </div>
        `;
        return;
    }

    // Sort results by similarity descending (highest first)
    const sortedResults = [...data.results].sort((a, b) => (b.similarity || 0) - (a.similarity || 0));

    // Count by type
    const emailCount = sortedResults.filter(r => r.result_type === 'email').length;
    const fileCount = sortedResults.filter(r => r.result_type === 'document').length;

    const resultsHtml = sortedResults.map((result, index) => {
        const similarity = result.similarity || 0;

        // Highlight top 3 results
        const isTopResult = index < 3;
        const topBadge = isTopResult ? `<span class="top-badge">Top ${index + 1}</span>` : '';

        if (result.result_type === 'email' && result.email) {
            return renderEmailItem(result.email, index, `unified-email-${index}`, similarity, topBadge, false);
        } else if (result.result_type === 'document' && result.document) {
            return renderDocumentItem(result.document, index, similarity, topBadge);
        }
        return '';
    }).join('');

    // Show stats about results
    const avgScore = sortedResults.reduce((sum, r) => sum + (r.similarity || 0), 0) / sortedResults.length;
    const topScore = sortedResults[0]?.similarity || 0;

    // Build type summary
    let typeSummary = '';
    if (emailCount > 0 && fileCount > 0) {
        typeSummary = `(${emailCount} emails, ${fileCount} files)`;
    } else if (emailCount > 0) {
        typeSummary = `(${emailCount} emails)`;
    } else if (fileCount > 0) {
        typeSummary = `(${fileCount} files)`;
    }

    container.innerHTML = `
        <div class="success-message">
            Found ${data.total || data.results.length} results for "${escapeHtml(data.query)}" ${typeSummary}
            <br>
            <small>Top similarity: ${(topScore * 100).toFixed(1)}% | Average: ${(avgScore * 100).toFixed(1)}%</small>
        </div>
        ${resultsHtml}
    `;
}

// Render a document search result item
function renderDocumentItem(doc, index, similarity, scoreBadge = '') {
    const docId = `doc-${index}`;

    // Get document info
    const title = doc.title || doc.original_filename || doc.filename || 'Untitled Document';
    const docType = doc.document_type || doc.mime_type || 'document';
    const summary = doc.summary || '';
    const previewText = doc.extracted_text_preview || summary || '';
    const previewShort = previewText.length > 200 ? previewText.substring(0, 200) + '...' : previewText;

    // Get origin info
    const origins = doc.origins || [];
    const primaryOrigin = origins[0] || {};
    const originPath = primaryOrigin.origin_path || '';
    const originHost = primaryOrigin.origin_host || '';
    const filename = originPath.split('/').pop() || title;

    // Format file size
    const fileSize = doc.file_size ? formatFileSize(doc.file_size) : '';

    // Document type icon
    const typeIcon = getDocumentTypeIcon(docType, doc.mime_type);

    const isTopResult = scoreBadge && scoreBadge.includes('Top');

    return `
        <div class="email-list-item document-item expandable ${isTopResult ? 'top-result' : ''}" id="doc-item-${doc.id}" data-doc-id="${doc.id}">
            <div class="email-list-header" onclick="toggleDocDetails('${docId}')">
                <div class="email-list-subject">
                    <span class="expand-icon" id="${docId}-icon">▶</span>
                    ${typeIcon} ${escapeHtml(title)}
                    ${scoreBadge ? `<span style="margin-left: 10px;">${scoreBadge}</span>` : ''}
                </div>
                <div class="email-list-date">${doc.first_seen_at ? formatDate(doc.first_seen_at) : (doc.created_at ? formatDate(doc.created_at) : '')}</div>
            </div>
            <div class="result-score" style="display: inline-block; margin-bottom: 8px;">${(similarity * 100).toFixed(1)}%</div>
            <div class="email-list-from" onclick="toggleDocDetails('${docId}')">📁 ${escapeHtml(filename)}${fileSize ? ` (${fileSize})` : ''}</div>
            ${previewShort ? `
                <div class="email-list-preview" onclick="toggleDocDetails('${docId}')">${escapeHtml(previewShort)}</div>
            ` : ''}
            <div class="result-tags" style="margin-top: 10px;">
                <span class="tag" style="background: #8b5cf6; color: white;">📄 File</span>
                ${docType ? `<span class="tag">${escapeHtml(docType)}</span>` : ''}
                ${originHost ? `<span class="tag" style="background: #10b981; color: white;">🖥️ ${escapeHtml(originHost)}</span>` : ''}
                ${doc.ai_category ? `<span class="tag">${escapeHtml(doc.ai_category)}</span>` : ''}
                ${doc.extraction_quality ? `<span class="tag">Quality: ${(doc.extraction_quality * 100).toFixed(0)}%</span>` : ''}
            </div>

            <div id="${docId}-details" class="email-details" style="display: none;">
                <div class="email-body">
                    ${summary ? `<p><strong>Summary:</strong> ${escapeHtml(summary)}</p>` : ''}
                    <p><strong>Dates:</strong></p>
                    <ul style="margin: 5px 0 10px 20px; padding: 0;">
                        ${doc.document_date ? `<li>File date: ${formatDate(doc.document_date)}</li>` : ''}
                        <li>First seen: ${doc.first_seen_at ? formatDate(doc.first_seen_at) : 'Unknown'}</li>
                        <li>Last seen: ${doc.last_seen_at ? formatDate(doc.last_seen_at) : 'Unknown'}</li>
                    </ul>
                    <p><strong>Locations (${origins.length}):</strong></p>
                    <ul style="margin: 5px 0 10px 20px; padding: 0;">
                        ${origins.map(o => {
                            const fullPath = o.origin_path || '';
                            const canOpenInFinder = fullPath.startsWith('/');
                            const finderUrl = canOpenInFinder ? `file://${encodeURI(fullPath)}` : '';
                            const modifiedAt = o.file_modified_at ? formatDate(o.file_modified_at) : '';
                            return `<li style="margin-bottom: 6px;">
                                ${o.origin_host ? `<strong>${escapeHtml(o.origin_host)}</strong>: ` : ''}
                                <code>${escapeHtml(fullPath || o.origin_filename || 'Unknown')}</code>
                                ${canOpenInFinder ? `<a href="${finderUrl}" class="btn-small" style="margin-left: 8px; padding: 2px 8px; font-size: 11px;" title="Open in Finder">📂 Open</a>` : ''}
                                ${modifiedAt ? `<br><span style="color: #666; font-size: 0.9em; margin-left: 20px;">Last edited: ${modifiedAt}</span>` : ''}
                            </li>`;
                        }).join('')}
                    </ul>
                    ${doc.checksum ? `<p><strong>Checksum:</strong> <code>${escapeHtml(doc.checksum.substring(0, 16))}...</code></p>` : ''}
                </div>
            </div>
        </div>
    `;
}

// Toggle document details expansion
function toggleDocDetails(docId) {
    const detailsEl = document.getElementById(`${docId}-details`);
    const iconEl = document.getElementById(`${docId}-icon`);

    if (detailsEl.style.display === 'none') {
        detailsEl.style.display = 'block';
        iconEl.textContent = '▼';
    } else {
        detailsEl.style.display = 'none';
        iconEl.textContent = '▶';
    }
}

// Get icon for document type
function getDocumentTypeIcon(docType, mimeType) {
    const type = (docType || mimeType || '').toLowerCase();
    if (type.includes('pdf')) return '📕';
    if (type.includes('word') || type.includes('doc')) return '📘';
    if (type.includes('excel') || type.includes('sheet') || type.includes('xls')) return '📗';
    if (type.includes('powerpoint') || type.includes('presentation') || type.includes('ppt')) return '📙';
    if (type.includes('image') || type.includes('png') || type.includes('jpg') || type.includes('jpeg')) return '🖼️';
    if (type.includes('text') || type.includes('txt')) return '📝';
    if (type.includes('csv')) return '📊';
    if (type.includes('zip') || type.includes('archive')) return '📦';
    return '📄';
}

// Format file size
function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return '';
    const units = ['B', 'KB', 'MB', 'GB'];
    let unitIndex = 0;
    let size = bytes;
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
}

// Reusable email item renderer
function renderEmailItem(email, index, idPrefix = 'email', scoreValue = null, scoreBadge = '', showWrongCategoryButton = true) {
    const metadata = email.email_metadata || {};
    const emailId = `${idPrefix}-${index}`;
    
    // Extract preview
    let preview = email.body_markdown || email.body_text || email.text_preview || '';
    let previewShort = preview.length > 200 ? preview.substring(0, 200) + '...' : preview;
    
    const currentCategory = metadata.ai_category || 'unknown';
    const escapedCategory = currentCategory.replace(/'/g, "\\'");
    
    // Check if it's a top result
    const isTopResult = scoreBadge && scoreBadge.includes('Top');
    
    return `
        <div class="email-list-item expandable ${isTopResult ? 'top-result' : ''}" id="email-item-${email.id}" data-email-id="${email.id}">
            <div class="email-list-header" onclick="toggleEmailDetails('${emailId}')">
                <div class="email-list-subject">
                    <span class="expand-icon" id="${emailId}-icon">▶</span>
                    ${escapeHtml(email.subject || 'No subject')}
                    ${scoreBadge ? `<span style="margin-left: 10px;">${scoreBadge}</span>` : ''}
                </div>
                <div class="email-list-date">${formatDate(email.date)}</div>
            </div>
            ${scoreValue !== null ? `
                <div class="result-score" style="display: inline-block; margin-bottom: 8px;">${(scoreValue * 100).toFixed(1)}%</div>
            ` : ''}
            <div class="email-list-from" onclick="toggleEmailDetails('${emailId}')">📧 ${escapeHtml(email.from_address || email.from || 'Unknown')}</div>
            ${previewShort ? `
                <div class="email-list-preview" onclick="toggleEmailDetails('${emailId}')">${escapeHtml(previewShort)}</div>
            ` : ''}
            <div class="result-tags" style="margin-top: 10px;">
                ${email.account_id ? `<span class="tag" style="background: #10b981; color: white;">📧 ${escapeHtml(email.account_id)}</span>` : ''}
                ${email.folder ? `<span class="tag" style="background: #6366f1; color: white;">📁 ${escapeHtml(email.folder)}</span>` : ''}
                ${metadata.awaiting_reply ? '<span class="tag tag-replied">✓ Replied</span>' : ''}
                ${metadata.needs_reply && !metadata.awaiting_reply ? '<span class="tag needs-reply">Needs Reply</span>' : ''}
                ${metadata.ai_category ? `<span class="tag">${escapeHtml(metadata.ai_category)}</span>` : ''}
                ${metadata.vip_level ? `<span class="tag">VIP: ${escapeHtml(metadata.vip_level)}</span>` : ''}
                ${metadata.ai_confidence ? `<span class="tag">Confidence: ${(metadata.ai_confidence * 100).toFixed(0)}%</span>` : ''}
                ${renderQuickScores(metadata, currentCategory)}
            </div>
            
            <div class="email-actions">
                ${email.message_id ? `
                    <a href="message://%3C${encodeURIComponent(email.message_id.replace(/^<|>$/g, ''))}%3E" 
                       class="action-btn btn-open-mail" 
                       onclick="event.stopPropagation();" 
                       title="Open in Apple Mail">
                        📬 Open in Mail
                    </a>
                    ${renderReplyButton(email, metadata)}
                ` : ''}
                ${showWrongCategoryButton ? `
                    <button class="action-btn btn-wrong-category" onclick="event.stopPropagation(); reportWrongCategory('${email.id}', '${escapedCategory}');">
                        ⚠️ Wrong Category
                    </button>
                ` : ''}
                <button class="action-btn btn-mark-spam" id="spam-btn-${email.id}" onclick="event.stopPropagation(); markAsSpam('${email.id}');">
                    🚫 Mark as Spam
                </button>
                <button class="action-btn btn-archive" id="archive-btn-${email.id}" onclick="event.stopPropagation(); archiveEmail('${email.id}');">
                    📦 Archive
                </button>
                <button class="action-btn btn-mark-handled" id="handled-btn-${email.id}" onclick="event.stopPropagation(); markAsHandled('${email.id}');">
                    ✓ Mark as Handled
                </button>
                <button class="action-btn btn-delete" id="delete-btn-${email.id}" onclick="event.stopPropagation(); deleteEmailToTrash('${email.id}');">
                    🗑️ Delete
                </button>
            </div>
            
            <div class="email-details" id="${emailId}" style="display: none;">
                <div class="email-detail-section">
                    <strong>Full Message:</strong>
                    <div class="email-body">${escapeHtml(preview).replace(/\n/g, '<br>')}</div>
                </div>
                
                ${metadata.ai_summary ? `
                    <div class="email-detail-section">
                        <strong>AI Summary:</strong>
                        <div>${escapeHtml(metadata.ai_summary)}</div>
                    </div>
                ` : ''}
                
                ${metadata.ai_reasoning ? `
                    <div class="email-detail-section">
                        <strong>Classification Reasoning:</strong>
                        <div>${escapeHtml(metadata.ai_reasoning)}</div>
                    </div>
                ` : ''}
                
                <div class="email-detail-section">
                    <strong>Email Details:</strong>
                    <div style="display: grid; grid-template-columns: auto 1fr; gap: 10px; font-size: 0.9rem;">
                        <span>Message ID:</span><span style="font-family: monospace; font-size: 0.8rem;">${escapeHtml(email.message_id || 'N/A')}</span>
                        <span>Folder:</span><span>${escapeHtml(email.folder || 'N/A')}</span>
                        <span>Has Attachments:</span><span>${email.has_attachments ? 'Yes' : 'No'}</span>
                        <span>Sentiment:</span><span>${escapeHtml(metadata.ai_sentiment || 'N/A')}</span>
                        <span>Urgency:</span><span>${escapeHtml(metadata.ai_urgency || 'N/A')} (${metadata.ai_urgency_score || 'N/A'}/10)</span>
                    </div>
                </div>
                
                <div class="email-detail-section">
                    <strong>Category-Specific Analysis:</strong>
                    <div style="display: grid; grid-template-columns: auto 1fr; gap: 10px; font-size: 0.9rem; background: #f8fafc; padding: 12px; border-radius: 6px;">
                        ${renderCategorySpecificData(metadata, currentCategory)}
                    </div>
                </div>
                
                <div class="email-detail-section">
                    <strong>Location History:</strong>
                    <div id="folder-history-${email.id}" style="font-size: 0.9rem;" data-email-id="${email.id}">
                        <div style="color: #666; padding: 10px;">⏳ Loading location history...</div>
                    </div>
                </div>
                
                ${email.account_id && accounts.length > 1 ? `
                    <div class="email-detail-section">
                        <strong>Move to Another Account:</strong>
                        <div style="display: flex; gap: 10px; margin-top: 10px; align-items: center;">
                            <select id="move-account-select-${email.id}" style="flex: 1; padding: 8px; border-radius: 4px; border: 1px solid #ddd;">
                                <option value="">Select account...</option>
                                ${accounts.filter(acc => {
                                    if (acc.nickname === email.account_id) return false;
                                    const currentAccount = accounts.find(a => a.nickname === email.account_id);
                                    return currentAccount && currentAccount.allow_moves_to && currentAccount.allow_moves_to.includes(acc.nickname);
                                }).map(acc => 
                                    `<option value="${escapeHtml(acc.nickname)}">${escapeHtml(acc.display_name)}</option>`
                                ).join('')}
                            </select>
                            <input type="text" id="move-folder-input-${email.id}" placeholder="INBOX" value="INBOX" style="width: 150px; padding: 8px; border-radius: 4px; border: 1px solid #ddd;">
                            <button class="action-btn" onclick="event.stopPropagation(); moveEmailToAccount('${email.id}');" style="padding: 8px 15px;">
                                📤 Move
                            </button>
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Helper function to render reply button with dropdown for answer options
function renderReplyButton(email, metadata) {
    const answerOptions = metadata.answer_options || [];
    
    // Debug logging
    console.log('[DEBUG] renderReplyButton called for email:', email.id);
    console.log('[DEBUG] Answer options:', answerOptions);
    
    // If no answer options, show simple reply button
    if (!answerOptions || answerOptions.length === 0) {
        return `
            <button class="action-btn btn-reply-mail" 
                    id="reply-btn-${email.id}" 
                    onclick="event.stopPropagation(); openReplyInMail('${email.id}', null);" 
                    title="Open reply all in Mail.app with Lorem Ipsum">
                💬 Reply All
            </button>
        `;
    }
    
    // Build dropdown menu with answer options
    const escapedId = email.id.replace(/'/g, "\\'");
    let dropdownItems = '';
    
    // Add "All Options" first
    const allOptionsText = answerOptions.map((opt, i) => 
        `[Option ${i+1}: ${opt.tone}]\\n${opt.text}\\n\\n`
    ).join('');
    
    dropdownItems += `
        <div class="reply-dropdown-item" onclick="event.stopPropagation(); openReplyInMail('${escapedId}', 'all');">
            📋 All Options (compare & delete)
        </div>
        <div class="reply-dropdown-divider"></div>
    `;
    
    // Add individual options
    answerOptions.forEach((option, index) => {
        const toneEmoji = option.tone === 'positive' ? '👍' : 
                         option.tone === 'decline' ? '👎' : 
                         option.tone === 'inquiry' ? '❓' : '💬';
        const toneLabel = option.tone.charAt(0).toUpperCase() + option.tone.slice(1);
        const preview = option.text.substring(0, 50) + (option.text.length > 50 ? '...' : '');
        
        dropdownItems += `
            <div class="reply-dropdown-item" 
                 onclick="event.stopPropagation(); openReplyInMail('${escapedId}', ${index});"
                 title="${escapeHtml(option.text)}">
                ${toneEmoji} ${toneLabel}
                <div class="reply-preview">${escapeHtml(preview)}</div>
            </div>
        `;
    });
    
    return `
        <div class="reply-button-container" onmouseenter="showReplyDropdown('${escapedId}')" onmouseleave="hideReplyDropdown('${escapedId}')">
            <button class="action-btn btn-reply-mail" 
                    id="reply-btn-${email.id}" 
                    onclick="event.stopPropagation(); openReplyInMail('${escapedId}', 0);" 
                    title="Click for first option, hover for all options">
                💬 Reply (${answerOptions.length})
            </button>
            <div class="reply-dropdown" id="reply-dropdown-${email.id}" style="display: none;">
                ${dropdownItems}
            </div>
        </div>
    `;
}

// Show/hide reply dropdown
function showReplyDropdown(emailId) {
    const dropdown = document.getElementById(`reply-dropdown-${emailId}`);
    if (dropdown) {
        dropdown.style.display = 'block';
    }
}

function hideReplyDropdown(emailId) {
    const dropdown = document.getElementById(`reply-dropdown-${emailId}`);
    if (dropdown) {
        // Small delay so click can register
        setTimeout(() => {
            dropdown.style.display = 'none';
        }, 200);
    }
}

// Helper function to render quick scores in collapsed view
function renderQuickScores(metadata, category) {
    const catData = metadata.category_specific_data || {};
    const scores = [];
    
    // Application categories - show key scores
    if (category.startsWith('application-')) {
        const sciScore = catData.scientific_excellence_score || metadata.scientific_excellence_score;
        const recScore = catData.recommendation_score || metadata.recommendation_score;
        
        if (sciScore) scores.push(`<span class="tag tag-score">📚 Excellence: ${sciScore}/10</span>`);
        if (recScore) {
            const color = recScore >= 7 ? '#10b981' : recScore >= 4 ? '#f59e0b' : '#ef4444';
            scores.push(`<span class="tag tag-score" style="background: ${color}; color: white;">⭐ Recommend: ${recScore}/10</span>`);
        }
    }
    
    // Invitation & Review categories - show relevance and prestige
    if (category.startsWith('invitation-') || category.startsWith('review-')) {
        if (metadata.relevance_score) scores.push(`<span class="tag tag-score">🎯 Relevance: ${metadata.relevance_score}/10</span>`);
        if (metadata.prestige_score) scores.push(`<span class="tag tag-score">✨ Prestige: ${metadata.prestige_score}/10</span>`);
        if (catData.time_commitment_hours) {
            const hours = catData.time_commitment_hours;
            const color = hours > 40 ? '#ef4444' : hours > 20 ? '#f59e0b' : '#10b981';
            scores.push(`<span class="tag tag-score" style="background: ${color}; color: white;">⏱️ ${hours}h</span>`);
        }
    }
    
    return scores.join('');
}

// Helper function to render category-specific data
function renderCategorySpecificData(metadata, category) {
    const catData = metadata.category_specific_data || {};
    const items = [];
    
    // Application categories - check both top-level and category_specific_data
    if (category.startsWith('application-')) {
        const appName = catData.applicant_name || metadata.applicant_name;
        const appInst = catData.applicant_institution || metadata.applicant_institution;
        const sciScore = catData.scientific_excellence_score || metadata.scientific_excellence_score;
        const sciReason = catData.scientific_excellence_reason || metadata.scientific_excellence_reason;
        const recScore = catData.recommendation_score || metadata.recommendation_score;
        const recReason = catData.recommendation_reason || metadata.recommendation_reason;
        
        if (appName) items.push(`<span>Applicant:</span><span><strong>${escapeHtml(appName)}</strong></span>`);
        if (appInst) items.push(`<span>Institution:</span><span>${escapeHtml(appInst)}</span>`);
        if (sciScore) items.push(`<span>Scientific Excellence:</span><span><strong>${sciScore}/10</strong><br><small>${sciReason ? escapeHtml(sciReason) : ''}</small></span>`);
        if (recScore) items.push(`<span>Recommendation:</span><span><strong>${recScore}/10</strong><br><small>${recReason ? escapeHtml(recReason) : ''}</small></span>`);
    }
    
    // Invitation & Review categories
    if (category.startsWith('invitation-') || category.startsWith('review-')) {
        if (catData.event_date) items.push(`<span>Event Date:</span><span><strong>${escapeHtml(catData.event_date)}</strong></span>`);
        if (catData.location) items.push(`<span>Location:</span><span>${escapeHtml(catData.location)}</span>`);
        if (metadata.relevance_score) items.push(`<span>Relevance:</span><span><strong>${metadata.relevance_score}/10</strong><br><small>${metadata.relevance_reason ? escapeHtml(metadata.relevance_reason) : ''}</small></span>`);
        if (metadata.prestige_score) items.push(`<span>Prestige:</span><span><strong>${metadata.prestige_score}/10</strong><br><small>${metadata.prestige_reason ? escapeHtml(metadata.prestige_reason) : ''}</small></span>`);
        if (catData.time_commitment_hours) items.push(`<span>Time Commitment:</span><span><strong>${catData.time_commitment_hours} hours</strong><br><small>${catData.time_commitment_reason ? escapeHtml(catData.time_commitment_reason) : ''}</small></span>`);
        if (metadata.deadline) items.push(`<span>Deadline:</span><span><strong style="color: #dc2626;">${escapeHtml(metadata.deadline)}</strong></span>`);
    }
    
    // Receipt categories
    if (category.startsWith('receipt-')) {
        if (catData.vendor) items.push(`<span>Vendor:</span><span>${escapeHtml(catData.vendor)}</span>`);
        if (catData.amount) items.push(`<span>Amount:</span><span><strong>${escapeHtml(catData.amount)} ${catData.currency || ''}</strong></span>`);
    }
    
    // Personal categories
    if (category.startsWith('personal-')) {
        if (catData.event_date) items.push(`<span>Event Date:</span><span>${escapeHtml(catData.event_date)}</span>`);
    }
    
    // Show suggested labels if any
    if (metadata.suggested_labels && metadata.suggested_labels.length > 0) {
        items.push(`<span>Suggested Labels:</span><span>${metadata.suggested_labels.map(l => '<span class="tag">' + escapeHtml(l) + '</span>').join(' ')}</span>`);
    }
    
    return items.length > 0 ? items.join('') : '<span style="grid-column: 1 / -1; color: #64748b;">No category-specific data available</span>';
}

// ============================================================================
// Browse by Category Tab
// ============================================================================

function initBrowse() {
    const browseButton = document.getElementById('browse-button');
    browseButton.addEventListener('click', loadCategoryEmails);
    
    // Load categories immediately on init, then auto-load "All"
    loadCategories().then(() => {
        // Auto-load "All" categories by default
        const categorySelect = document.getElementById('category-select');
        categorySelect.value = '_all';
        loadCategoryEmails();
    });
}

async function loadCategories() {
    const categorySelect = document.getElementById('category-select');
    
    try {
        const response = await fetch(`${API_BASE}/api/stats`, FETCH_OPTIONS);
        if (!response.ok) {
            throw new Error('Failed to load categories');
        }
        
        const stats = await response.json();  // Fixed: added await
        const categories = stats.top_categories || stats.categories_breakdown || {};
        
        console.log('Categories loaded:', categories);
        
        if (!categories || Object.keys(categories).length === 0) {
            categorySelect.innerHTML = '<option value="">No categories found</option>';
            return;
        }
        
        // Define category priority order (most urgent/important first)
        const categoryPriority = {
            // Urgent/Action Required
            'work-urgent': 1,
            'work-admin': 2,
            'application-deadline': 3,
            'review-promotion': 13,
            'review-peer-journal': 5,
            'review-peer-conference': 6,
            
            // Important Work
            'work-scheduling': 7,
            'work-colleague': 8,
            'work-student': 9,
            'application-phd': 10,
            'application-postdoc': 11,
            'application-visiting': 12,
            'application-bsc-msc-thesis': 13,
            
            // Invitations (Important but less urgent)
            'invitation-speaking': 4,
            'invitation-collaboration': 15,
            'invitation-editorial': 16,
            'invitation-event': 17,
            
            // Applications & Personal
            'application-intern': 18,
            'application-other': 19,
            'personal-family': 20,
            'personal-transaction': 21,
            'personal-other': 22,
            
            // Receipts & Notifications
            'receipt-travel': 23,
            'receipt-subscription': 24,
            'receipt-online': 25,
            'notification-calendar': 26,
            'notification-technical': 27,
            'notification-other': 28,
            
            // Low Priority
            'work-no-action-needed': 29,
            'newsletter-scientific': 30,
            'newsletter-general': 31,
            'marketing': 32,
            'work-other': 33,
            'spam': 34
        };
        
        // Sort categories by priority (lower number = higher priority)
        const sortedCategories = Object.entries(categories)
            .sort((a, b) => {
                const priorityA = categoryPriority[a[0]] || 999;
                const priorityB = categoryPriority[b[0]] || 999;
                // First sort by priority, then by count if same priority
                if (priorityA === priorityB) {
                    return b[1] - a[1]; // Higher count first
                }
                return priorityA - priorityB; // Lower priority number first
            });
        
        // Add "All" option at the beginning
        const totalCount = sortedCategories.reduce((sum, [_, count]) => sum + count, 0);
        const allOption = `<option value="_all">All Categories (${totalCount})</option>`;
        
        categorySelect.innerHTML = allOption + sortedCategories.map(([category, count]) => 
            `<option value="${escapeHtml(category)}">${escapeHtml(category)} (${count})</option>`
        ).join('');
        
    } catch (error) {
        console.error('Failed to load categories:', error);
        categorySelect.innerHTML = '<option value="">Error loading categories</option>';
    }
}

async function loadCategoryEmails() {
    const category = document.getElementById('category-select').value;
    const limit = parseInt(document.getElementById('browse-limit').value);
    const hideHandled = document.getElementById('hide-handled')?.checked ?? true;
    const resultsEl = document.getElementById('browse-results');
    const browseButton = document.getElementById('browse-button');
    
    if (!category) {
        showError(resultsEl, 'Please select a category');
        return;
    }
    
    browseButton.disabled = true;
    browseButton.textContent = 'Loading...';
    resultsEl.innerHTML = '<div class="loading">Loading emails</div>';
    
    try {
        if (category === '_all') {
            // For "All", we need to fetch emails from multiple categories
            await loadAllCategoriesEmails(limit, hideHandled, resultsEl, selectedAccount);
        } else {
            // Single category
            let url = `${API_BASE}/api/emails?category=${encodeURIComponent(category)}&page_size=${limit}&sort_by=date&sort_order=desc`;
            
            // Add account filter if selected
            if (selectedAccount) {
                url += `&account=${encodeURIComponent(selectedAccount)}`;
            }
            
            // Add exclude_handled parameter if checkbox is checked
            if (hideHandled) {
                url += '&exclude_handled=true';
            }
            
            const response = await fetch(url, FETCH_OPTIONS);
            
            if (!response.ok) {
                throw new Error(`Failed to load emails: ${response.statusText}`);
            }
            
            const data = await response.json();
            displayCategoryEmails(data, category, resultsEl);
        }
        
    } catch (error) {
        showError(resultsEl, `Failed to load emails: ${error.message}`);
    } finally {
        browseButton.disabled = false;
        browseButton.textContent = 'Load Emails';
    }
}

async function loadAllCategoriesEmails(limit, hideHandled, container, accountFilter = null) {
    // Define category priority (same as in loadCategories)
    const categoryPriority = {
        'work-urgent': 1,
        'work-admin': 2,
        'application-deadline': 3,
        'invitation-speaking': 4,
        'review-peer-journal': 5,
        'review-peer-conference': 6,
        'work-scheduling': 7,
        'work-colleague': 8,
        'work-student': 9,
        'application-phd': 10,
        'application-postdoc': 11,
        'application-visiting': 12,
        'application-bsc-msc-thesis': 13,
        'review-promotion': 14,
        'invitation-collaboration': 15,
        'invitation-editorial': 16,
        'invitation-event': 17,
        'application-intern': 18,
        'application-other': 19,
        'personal-family': 20,
        'personal-transaction': 21,
        'personal-other': 22,
        'receipt-travel': 23,
        'receipt-subscription': 24,
        'receipt-online': 25,
        'notification-calendar': 26,
        'notification-technical': 27,
        'notification-other': 28,
        'work-no-action-needed': 29,
        'newsletter-scientific': 30,
        'newsletter-general': 31,
        'marketing': 32,
        'work-other': 33,
        'spam': 34
    };
    
    // Fetch all emails (no category filter)
    let url = `${API_BASE}/api/emails?page_size=${limit}&sort_by=date&sort_order=desc`;
    
    // Add account filter if selected
    if (accountFilter) {
        url += `&account=${encodeURIComponent(accountFilter)}`;
    }
    
    if (hideHandled) {
        url += '&exclude_handled=true';
    }
    
    const response = await fetch(url, FETCH_OPTIONS);
    if (!response.ok) {
        throw new Error(`Failed to load emails: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    if (!data.emails || data.emails.length === 0) {
        container.innerHTML = '<div class="error-message">No emails found.</div>';
        return;
    }
    
    // Sort emails by category priority, then by date
    const sortedEmails = data.emails.sort((a, b) => {
        const categoryA = a.email_metadata?.ai_category || 'unknown';
        const categoryB = b.email_metadata?.ai_category || 'unknown';
        const priorityA = categoryPriority[categoryA] || 999;
        const priorityB = categoryPriority[categoryB] || 999;
        
        // First sort by category priority
        if (priorityA !== priorityB) {
            return priorityA - priorityB;
        }
        // Then by date (most recent first)
        return new Date(b.date) - new Date(a.date);
    });
    
    const emailsHtml = sortedEmails.map((email, index) => {
        return renderEmailItem(email, index, `browse-${index}`);
    }).join('');
    
    container.innerHTML = `
        <div class="category-header">
            <h3>All Categories (by Importance)</h3>
            <div class="email-count-badge">${data.total} total emails</div>
        </div>
        ${emailsHtml}
        ${data.total > sortedEmails.length ? `
            <div class="success-message">
                Showing ${sortedEmails.length} of ${data.total} emails. 
                Increase "Emails per page" to see more.
            </div>
        ` : ''}
    `;
}

function displayCategoryEmails(data, category, container) {
    if (!data.emails || data.emails.length === 0) {
        container.innerHTML = '<div class="error-message">No emails found in this category.</div>';
        return;
    }
    
    console.log('Displaying category emails:', data);
    
    const emailsHtml = data.emails.map((email, index) => {
        return renderEmailItem(email, index, `browse-${index}`);
    }).join('');
    
    container.innerHTML = `
        <div class="category-header">
            <h3>${escapeHtml(category)}</h3>
            <div class="email-count-badge">${data.total} total emails</div>
        </div>
        ${emailsHtml}
        ${data.total > data.emails.length ? `
            <div class="success-message">
                Showing ${data.emails.length} of ${data.total} emails. 
                Increase "Emails per page" to see more.
            </div>
        ` : ''}
    `;
}

// ============================================================================
// Costs Tab
// ============================================================================

function initCosts() {
    const refreshButton = document.getElementById('refresh-costs-button');
    refreshButton.addEventListener('click', loadCosts);
}

async function loadCosts() {
    const days = parseInt(document.getElementById('cost-days').value);
    const detailsEl = document.getElementById('cost-details');
    
    detailsEl.innerHTML = '<div class="loading">Loading cost data</div>';
    
    try {
        // Load summary
        const summaryResponse = await fetch(`${API_BASE}/api/costs/summary`, FETCH_OPTIONS);
        if (summaryResponse.ok) {
            const summary = await summaryResponse.json();
            document.getElementById('cost-today').textContent = `$${summary.today.toFixed(2)}`;
            document.getElementById('cost-month').textContent = `$${summary.this_month.toFixed(2)}`;
            document.getElementById('cost-total').textContent = `$${summary.total.toFixed(2)}`;
        }
        
        // Load overview
        const overviewResponse = await fetch(`${API_BASE}/api/costs/overview?days=${days}`, FETCH_OPTIONS);
        if (overviewResponse.ok) {
            const overview = await overviewResponse.json();
            console.log('Cost overview data:', overview);
            displayCostDetails(overview, detailsEl);
        } else {
            const errorText = await overviewResponse.text();
            console.error('Cost overview error:', errorText);
            detailsEl.innerHTML = `<div class="error-message">Cost overview not available (${overviewResponse.status}). Cost tracking endpoints may not be enabled.</div>`;
        }
        
    } catch (error) {
        console.error('Load costs error:', error);
        showError(detailsEl, `Failed to load costs: ${error.message}`);
    }
}

function displayCostDetails(data, container) {
    console.log('Displaying cost details:', data);
    
    if (data.note || data.message) {
        container.innerHTML = `<div class="error-message">${escapeHtml(data.note || data.message)}</div>`;
        return;
    }
    
    // Build HTML sections
    let html = `
        <div class="cost-section">
            <h3>📊 Period Summary (${data.period_days || 30} days)</h3>
            <div class="cost-item">
                <span class="cost-item-name">Total Cost</span>
                <span class="cost-item-value">$${(data.total_cost || 0).toFixed(2)}</span>
            </div>
            <div class="cost-item">
                <span class="cost-item-name">Total Tokens</span>
                <span class="cost-item-value">${(data.total_tokens || 0).toLocaleString()}</span>
            </div>
            <div class="cost-item">
                <span class="cost-item-name">Avg Daily Cost</span>
                <span class="cost-item-value">$${(data.avg_daily_cost || 0).toFixed(2)}</span>
            </div>
        </div>
    `;
    
    // Add projections if available
    if (data.projections) {
        html += `
            <div class="cost-section">
                <h3>📈 Projections</h3>
                <div class="cost-item">
                    <span class="cost-item-name">Projected Monthly</span>
                    <span class="cost-item-value">$${(data.projections.monthly || 0).toFixed(2)}</span>
                </div>
                <div class="cost-item">
                    <span class="cost-item-name">Projected Yearly</span>
                    <span class="cost-item-value">$${(data.projections.yearly || 0).toFixed(2)}</span>
                </div>
            </div>
        `;
    }
    
    // Add model breakdown if available
    if (data.breakdown_by_model && data.breakdown_by_model.length > 0) {
        html += `
            <div class="cost-section">
                <h3>🤖 Cost by Model</h3>
                ${data.breakdown_by_model.map(item => `
                    <div class="cost-item">
                        <span class="cost-item-name">${escapeHtml(item.model)} (${item.calls.toLocaleString()} calls)</span>
                        <span class="cost-item-value">$${item.cost.toFixed(2)}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    // Add task breakdown if available
    if (data.breakdown_by_task && data.breakdown_by_task.length > 0) {
        html += `
            <div class="cost-section">
                <h3>⚙️ Cost by Task</h3>
                ${data.breakdown_by_task.map(item => `
                    <div class="cost-item">
                        <span class="cost-item-name">${escapeHtml(item.task)}</span>
                        <span class="cost-item-value">$${item.cost.toFixed(2)}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    container.innerHTML = html;
}

// ============================================================================
// Process Tab
// ============================================================================

function initProcess() {
    const processButton = document.getElementById('process-button');
    processButton.addEventListener('click', triggerProcessing);
    
    // Disable limit input when "new only" is checked
    const newOnlyCheckbox = document.getElementById('process-new-only');
    const limitInput = document.getElementById('process-limit');
    
    newOnlyCheckbox.addEventListener('change', () => {
        limitInput.disabled = newOnlyCheckbox.checked;
    });
}

async function triggerProcessing() {
    const newOnly = document.getElementById('process-new-only').checked;
    const limit = newOnly ? null : parseInt(document.getElementById('process-limit').value);
    const dryRun = document.getElementById('process-dry-run').checked;
    const useAI = document.getElementById('process-use-ai').checked;
    const embeddings = document.getElementById('process-embeddings').checked;
    
    const processButton = document.getElementById('process-button');
    const statusEl = document.getElementById('process-status');
    
    processButton.disabled = true;
    processButton.textContent = 'Starting...';
    
    try {
        const response = await fetch(`${API_BASE}/api/process/trigger`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                limit: limit,
                new_only: newOnly,
                dry_run: dryRun,
                use_ai: useAI,
                generate_embeddings: embeddings
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start processing');
        }
        
        const data = await response.json();
        
        statusEl.classList.add('visible', 'running');
        statusEl.innerHTML = `
            <h3>⚙️ Processing Started</h3>
            <p>${data.message}</p>
            <div class="process-progress">Checking status...</div>
        `;
        
        // Start polling for status
        startProcessingStatusCheck();
        
    } catch (error) {
        statusEl.classList.add('visible', 'error');
        statusEl.innerHTML = `
            <h3>❌ Error</h3>
            <p>${error.message}</p>
        `;
        processButton.disabled = false;
        processButton.textContent = 'Start Processing';
    }
}

function startProcessingStatusCheck() {
    if (processingCheckInterval) {
        clearInterval(processingCheckInterval);
    }
    
    processingCheckInterval = setInterval(checkProcessingStatus, 2000);
    checkProcessingStatus();
}

async function checkProcessingStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/process/status`, FETCH_OPTIONS);
        const status = await response.json();
        
        const statusEl = document.getElementById('process-status');
        const processButton = document.getElementById('process-button');
        
        if (!statusEl.classList.contains('visible')) {
            return;
        }
        
        if (status.is_running) {
            statusEl.className = 'process-status visible running';
            statusEl.innerHTML = `
                <h3>⚙️ Processing Running</h3>
                <p>Status: ${escapeHtml(status.status)}</p>
                ${status.progress ? `<div class="process-progress">${escapeHtml(status.progress)}</div>` : ''}
            `;
        } else if (status.status === 'completed') {
            statusEl.className = 'process-status visible completed';
            statusEl.innerHTML = `
                <h3>✅ Processing Completed</h3>
                ${status.progress ? `<div class="process-progress">${escapeHtml(status.progress)}</div>` : ''}
            `;
            clearInterval(processingCheckInterval);
            processButton.disabled = false;
            processButton.textContent = 'Start Processing';
        } else if (status.status === 'failed' || status.status === 'error') {
            statusEl.className = 'process-status visible error';
            statusEl.innerHTML = `
                <h3>❌ Processing Failed</h3>
                <p>${escapeHtml(status.error || 'Unknown error')}</p>
            `;
            clearInterval(processingCheckInterval);
            processButton.disabled = false;
            processButton.textContent = 'Start Processing';
        } else {
            processButton.disabled = false;
            processButton.textContent = 'Start Processing';
        }
        
    } catch (error) {
        console.error('Failed to check processing status:', error);
    }
}

// ============================================================================
// Stats Tab
// ============================================================================

function initStats() {
    const refreshButton = document.getElementById('refresh-stats-button');
    refreshButton.addEventListener('click', loadStats);
}

function initStatsSubtabs() {
    // Initialize sub-tabs for categories vs folders
    // This is called when switching to stats tab to ensure DOM elements exist
    const subtabButtons = document.querySelectorAll('.stats-subtab-button');
    
    // Remove old listeners if any by cloning and replacing
    subtabButtons.forEach(button => {
        const newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);
    });
    
    // Add fresh event listeners
    document.querySelectorAll('.stats-subtab-button').forEach(button => {
        button.addEventListener('click', () => {
            const subtab = button.dataset.subtab;
            switchStatsSubtab(subtab);
        });
    });
}

function switchStatsSubtab(subtab) {
    // Update button states
    document.querySelectorAll('.stats-subtab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-subtab="${subtab}"]`).classList.add('active');
    
    // Show/hide appropriate breakdown
    const categoryBreakdown = document.getElementById('category-breakdown');
    const folderBreakdown = document.getElementById('folder-breakdown');
    
    if (subtab === 'categories') {
        categoryBreakdown.style.display = 'block';
        folderBreakdown.style.display = 'none';
    } else if (subtab === 'folders') {
        categoryBreakdown.style.display = 'none';
        folderBreakdown.style.display = 'block';
    }
}

async function loadStats() {
    const refreshButton = document.getElementById('refresh-stats-button');
    refreshButton.disabled = true;
    refreshButton.textContent = 'Loading...';
    
    try {
        const response = await fetch(`${API_BASE}/api/stats`, FETCH_OPTIONS);
        if (!response.ok) {
            throw new Error('Failed to load stats');
        }
        
        const stats = await response.json();
        
        // Update stat cards
        document.getElementById('stat-total').textContent = stats.total_emails?.toLocaleString() || '0';
        document.getElementById('stat-embeddings').textContent = stats.with_embeddings?.toLocaleString() || '0';
        document.getElementById('stat-ai').textContent = stats.ai_classifications?.toLocaleString() || '0';
        document.getElementById('stat-reply').textContent = stats.needs_reply_count?.toLocaleString() || stats.needs_reply?.toLocaleString() || '0';
        
        // Display category breakdown
        if (stats.top_categories || stats.categories_breakdown) {
            const categories = stats.top_categories || stats.categories_breakdown;
            displayCategories(categories);
        }
        
        // Display folder breakdown
        if (stats.folders_breakdown) {
            displayFolders(stats.folders_breakdown);
        }
        
    } catch (error) {
        const breakdownEl = document.getElementById('category-breakdown');
        showError(breakdownEl, `Failed to load stats: ${error.message}`);
    } finally {
        refreshButton.disabled = false;
        refreshButton.textContent = 'Refresh Stats';
    }
}

function displayCategories(categories) {
    const container = document.getElementById('category-breakdown');
    
    if (!categories || Object.keys(categories).length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #666;">No categories found</p>';
        return;
    }
    
    // Sort by count
    const sorted = Object.entries(categories).sort((a, b) => b[1] - a[1]);
    
    const html = sorted.map(([category, count]) => `
        <div class="category-item" onclick="switchToCategoryBrowse('${category.replace(/'/g, "\\'")}')">
            <span class="category-name">${escapeHtml(category)}</span>
            <span class="category-count">${count.toLocaleString()}</span>
        </div>
    `).join('');
    
    container.innerHTML = html;
}

function displayFolders(folders) {
    const container = document.getElementById('folder-breakdown');
    
    if (!folders || Object.keys(folders).length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #666;">No folders found</p>';
        return;
    }
    
    // Build hierarchical tree from flat folder list
    const tree = buildFolderTree(folders);
    
    // Render the tree
    const html = renderFolderTree(tree, 0);
    container.innerHTML = html;
}

function buildFolderTree(folders) {
    // Convert flat folder list into hierarchical structure
    // Folders can be separated by '/' or '.'
    const tree = {};
    
    for (const [folderPath, count] of Object.entries(folders)) {
        // Determine separator (typically '/' for IMAP, but could be '.')
        const separator = folderPath.includes('/') ? '/' : '.';
        const parts = folderPath.split(separator);
        
        let current = tree;
        for (let i = 0; i < parts.length; i++) {
            const part = parts[i];
            
            if (!current[part]) {
                current[part] = {
                    name: part,
                    fullPath: parts.slice(0, i + 1).join(separator),
                    count: 0,
                    children: {}
                };
            }
            
            // If this is the final part, set the count
            if (i === parts.length - 1) {
                current[part].count = count;
            }
            
            current = current[part].children;
        }
    }
    
    return tree;
}

function renderFolderTree(tree, depth) {
    if (!tree || Object.keys(tree).length === 0) {
        return '';
    }
    
    // Sort folders alphabetically
    const sortedFolders = Object.entries(tree).sort(([a], [b]) => a.localeCompare(b));
    
    let html = '';
    for (const [folderName, folderData] of sortedFolders) {
        const hasChildren = Object.keys(folderData.children).length > 0;
        const indent = depth * 20; // 20px per level
        const expandIcon = hasChildren ? '▸' : ''; // Arrow for expandable folders
        
        // Create folder item
        html += `
            <div class="folder-item" data-depth="${depth}" data-path="${escapeHtml(folderData.fullPath)}">
                <div class="folder-row" style="padding-left: ${indent}px;" onclick="toggleFolderExpand(this)">
                    <span class="folder-expand">${expandIcon}</span>
                    <span class="folder-icon">📁</span>
                    <span class="folder-name">${escapeHtml(folderName)}</span>
                    <span class="folder-count">${folderData.count.toLocaleString()}</span>
                </div>
                <div class="folder-children" style="display: none;">
                    ${renderFolderTree(folderData.children, depth + 1)}
                </div>
            </div>
        `;
    }
    
    return html;
}

function toggleFolderExpand(element) {
    const folderItem = element.closest('.folder-item');
    const childrenDiv = folderItem.querySelector('.folder-children');
    const expandIcon = element.querySelector('.folder-expand');
    
    if (childrenDiv && childrenDiv.innerHTML.trim() !== '') {
        const isExpanded = childrenDiv.style.display !== 'none';
        
        if (isExpanded) {
            childrenDiv.style.display = 'none';
            expandIcon.textContent = '▸';
        } else {
            childrenDiv.style.display = 'block';
            expandIcon.textContent = '▾';
        }
    }
}

function switchToCategoryBrowse(category) {
    // Switch to browse tab
    switchTab('browse');
    
    // Wait a moment for tab to load
    setTimeout(() => {
        // Select the category
        const categorySelect = document.getElementById('category-select');
        categorySelect.value = category;
        
        // Load emails
        loadCategoryEmails();
    }, 100);
}

// ============================================================================
// Utility Functions
// ============================================================================

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateString) {
    if (!dateString) return 'Unknown date';

    const date = new Date(dateString);
    const now = new Date();

    // Compare by calendar date (ignoring time)
    const dateOnly = new Date(date.getFullYear(), date.getMonth(), date.getDate());
    const nowOnly = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const diffDays = Math.round((nowOnly - dateOnly) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
        return 'Today';
    } else if (diffDays === 1) {
        return 'Yesterday';
    } else if (diffDays < 7) {
        return `${diffDays} days ago`;
    } else {
        return date.toLocaleDateString();
    }
}

function showError(container, message) {
    container.innerHTML = `<div class="error-message">${escapeHtml(message)}</div>`;
}

function toggleEmailDetails(emailId) {
    const detailsEl = document.getElementById(emailId);
    const iconEl = document.getElementById(emailId + '-icon');
    
    if (detailsEl.style.display === 'none') {
        detailsEl.style.display = 'block';
        iconEl.textContent = '▼';
        
        // Auto-load folder history when expanding email
        const folderHistoryContainer = detailsEl.querySelector('[id^="folder-history-"]');
        if (folderHistoryContainer) {
            const actualEmailId = folderHistoryContainer.getAttribute('data-email-id');
            if (actualEmailId) {
                // Only load if not already loaded
                const currentContent = folderHistoryContainer.innerHTML;
                if (currentContent.includes('Loading folder history')) {
                    loadFolderHistory(actualEmailId);
                }
            }
        }
    } else {
        detailsEl.style.display = 'none';
        iconEl.textContent = '▶';
    }
}

// Global state for category modal
let categoryModalState = {
    emailId: null,
    currentCategory: null
};

async function reportWrongCategory(emailId, currentCategory) {
    // Store state
    categoryModalState.emailId = emailId;
    categoryModalState.currentCategory = currentCategory;
    
    // Update modal text
    document.getElementById('current-category-text').textContent = currentCategory;
    
    // Load categories into dropdown
    await loadCategoriesForModal();
    
    // Show modal
    document.getElementById('category-modal').classList.add('active');
}

async function loadCategoriesForModal() {
    const select = document.getElementById('suggested-category-select');
    
    try {
        const response = await fetch(`${API_BASE}/api/stats`, FETCH_OPTIONS);
        if (response.ok) {
            const stats = await response.json();
            const categories = stats.categories_breakdown || {};
            
            // Sort categories alphabetically
            const sortedCategories = Object.keys(categories).sort();
            
            select.innerHTML = `
                <option value="">Select a category...</option>
                ${sortedCategories.map(cat => 
                    `<option value="${escapeHtml(cat)}">${escapeHtml(cat)}</option>`
                ).join('')}
                <option value="__other__">Other (specify custom)</option>
            `;
        }
    } catch (error) {
        console.error('Failed to load categories:', error);
        select.innerHTML = '<option value="">Error loading categories</option>';
    }
    
    // Listen for "Other" selection
    select.addEventListener('change', function() {
        const customGroup = document.getElementById('custom-category-group');
        if (this.value === '__other__') {
            customGroup.style.display = 'block';
        } else {
            customGroup.style.display = 'none';
        }
    });
}

function closeCategoryModal() {
    document.getElementById('category-modal').classList.remove('active');
    document.getElementById('custom-category-input').value = '';
    document.getElementById('suggested-category-select').value = '';
    document.getElementById('custom-category-group').style.display = 'none';
}

async function submitCategoryFeedback() {
    const emailId = categoryModalState.emailId;
    const currentCategory = categoryModalState.currentCategory;
    const selectedCategory = document.getElementById('suggested-category-select').value;
    const customCategory = document.getElementById('custom-category-input').value.trim();
    
    // Determine the suggested category
    let suggestedCategory = '';
    if (selectedCategory === '__other__' && customCategory) {
        suggestedCategory = customCategory;
    } else if (selectedCategory && selectedCategory !== '__other__') {
        suggestedCategory = selectedCategory;
    }
    
    closeCategoryModal();
    
    try {
        const notes = suggestedCategory 
            ? `Wrong category: was "${currentCategory}", should be "${suggestedCategory}". Reported ${new Date().toISOString().split('T')[0]}`
            : `Wrong category: should not be "${currentCategory}". Reported ${new Date().toISOString().split('T')[0]}`;
        
        const response = await fetch(`${API_BASE}/api/emails/${emailId}/metadata`, {
            method: 'PUT',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_notes: notes,
                project_tags: ['wrong-category', 'needs-review']
            })
        });
        
        console.log('Wrong category response:', response.status);
        
        if (response.ok) {
            const message = suggestedCategory 
                ? `✓ Wrong category (suggested: ${suggestedCategory})`
                : '✓ Marked as wrong category';
            removeEmailFromList(emailId, message);
        } else {
            const errorText = await response.text();
            alert(`Failed to record feedback: ${errorText}`);
        }
    } catch (error) {
        console.error('Error reporting wrong category:', error);
        alert('Error: ' + error.message);
    }
}

// Helper function to set button loading state
function setButtonLoading(buttonId, loading, originalText = null) {
    const button = document.getElementById(buttonId);
    if (!button) return;
    
    if (loading) {
        button.dataset.originalText = button.textContent;
        button.textContent = '⏳ Processing...';
        button.disabled = true;
        button.style.opacity = '0.6';
    } else {
        button.textContent = originalText || button.dataset.originalText || button.textContent;
        button.disabled = false;
        button.style.opacity = '1';
        delete button.dataset.originalText;
    }
}

// Show error dialog with choices
function showErrorDialog(title, message, onRetry, onCancel) {
    const existingDialog = document.getElementById('error-dialog');
    if (existingDialog) {
        existingDialog.remove();
    }
    
    const dialog = document.createElement('div');
    dialog.id = 'error-dialog';
    dialog.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        z-index: 10000;
        max-width: 500px;
        width: 90%;
    `;
    
    dialog.innerHTML = `
        <div style="margin-bottom: 20px;">
            <h3 style="margin: 0 0 15px 0; color: #ef4444; font-size: 1.3rem;">❌ ${escapeHtml(title)}</h3>
            <p style="margin: 0; color: #666; line-height: 1.6;">${escapeHtml(message)}</p>
        </div>
        <div style="display: flex; gap: 10px; justify-content: flex-end;">
            <button id="error-cancel-btn" style="padding: 10px 20px; border: 1px solid #ddd; background: white; border-radius: 6px; cursor: pointer; font-size: 0.95rem;">
                Cancel
            </button>
            <button id="error-retry-btn" style="padding: 10px 20px; border: none; background: #3b82f6; color: white; border-radius: 6px; cursor: pointer; font-size: 0.95rem;">
                Retry
            </button>
        </div>
    `;
    
    // Add backdrop
    const backdrop = document.createElement('div');
    backdrop.id = 'error-backdrop';
    backdrop.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        z-index: 9999;
    `;
    
    document.body.appendChild(backdrop);
    document.body.appendChild(dialog);
    
    document.getElementById('error-cancel-btn').onclick = () => {
        backdrop.remove();
        dialog.remove();
        if (onCancel) onCancel();
    };
    
    document.getElementById('error-retry-btn').onclick = () => {
        backdrop.remove();
        dialog.remove();
        if (onRetry) onRetry();
    };
}

async function markAsHandled(emailId) {
    const buttonId = `handled-btn-${emailId}`;
    setButtonLoading(buttonId, true);
    
    try {
        const response = await fetch(`${API_BASE}/api/emails/${emailId}/mark-handled`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        console.log('Mark handled response:', data);
        
        if (response.ok && data.success) {
            // Show success briefly, then fade out (but don't remove - stays in folder)
            setButtonLoading(buttonId, false, '✅ Handled!');
            setTimeout(() => {
                setButtonLoading(buttonId, false, '✓ Mark as Handled');
            }, 2000);
        } else {
            setButtonLoading(buttonId, false);
            const errorMsg = data.error || data.message || 'Unknown error';
            showErrorDialog(
                'Failed to Mark as Handled',
                `IMAP operation failed: ${errorMsg}`,
                () => markAsHandled(emailId),
                null
            );
        }
    } catch (error) {
        console.error('Error marking as handled:', error);
        setButtonLoading(buttonId, false);
        showErrorDialog(
            'Network Error',
            `Could not connect to server: ${error.message}`,
            () => markAsHandled(emailId),
            null
        );
    }
}

async function markAsSpam(emailId) {
    const buttonId = `spam-btn-${emailId}`;
    setButtonLoading(buttonId, true);
    
    try {
        const response = await fetch(`${API_BASE}/api/emails/${emailId}/mark-spam`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        console.log('Mark spam response:', data);
        
        if (response.ok && data.success) {
            removeEmailFromList(emailId, `🚫 Moved to ${data.new_folder || 'Spam'}`);
        } else {
            setButtonLoading(buttonId, false);
            const errorMsg = data.error || data.message || 'Unknown error';
            showErrorDialog(
                'Failed to Mark as Spam',
                `IMAP operation failed: ${errorMsg}`,
                () => markAsSpam(emailId),
                null
            );
        }
    } catch (error) {
        console.error('Error marking as spam:', error);
        setButtonLoading(buttonId, false);
        showErrorDialog(
            'Network Error',
            `Could not connect to server: ${error.message}`,
            () => markAsSpam(emailId),
            null
        );
    }
}

async function archiveEmail(emailId) {
    const buttonId = `archive-btn-${emailId}`;
    setButtonLoading(buttonId, true);
    
    try {
        const response = await fetch(`${API_BASE}/api/emails/${emailId}/archive`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        console.log('Archive response:', data);
        
        if (response.ok && data.success) {
            removeEmailFromList(emailId, `📦 Moved to ${data.new_folder || 'Archive'}`);
        } else {
            setButtonLoading(buttonId, false);
            const errorMsg = data.error || data.message || 'Unknown error';
            showErrorDialog(
                'Failed to Archive',
                `IMAP operation failed: ${errorMsg}`,
                () => archiveEmail(emailId),
                null
            );
        }
    } catch (error) {
        console.error('Error archiving email:', error);
        setButtonLoading(buttonId, false);
        showErrorDialog(
            'Network Error',
            `Could not connect to server: ${error.message}`,
            () => archiveEmail(emailId),
            null
        );
    }
}

async function deleteEmailToTrash(emailId) {
    const buttonId = `delete-btn-${emailId}`;
    setButtonLoading(buttonId, true);
    
    try {
        const response = await fetch(`${API_BASE}/api/emails/${emailId}/delete`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        console.log('Delete response:', data);
        
        if (response.ok && data.success) {
            removeEmailFromList(emailId, `🗑️ Moved to ${data.new_folder || 'Trash'}`);
        } else {
            setButtonLoading(buttonId, false);
            const errorMsg = data.error || data.message || 'Unknown error';
            showErrorDialog(
                'Failed to Delete',
                `IMAP operation failed: ${errorMsg}`,
                () => deleteEmailToTrash(emailId),
                null
            );
        }
    } catch (error) {
        console.error('Error deleting email:', error);
        setButtonLoading(buttonId, false);
        showErrorDialog(
            'Network Error',
            `Could not connect to server: ${error.message}`,
            () => deleteEmailToTrash(emailId),
            null
        );
    }
}

function removeEmailFromList(emailId, message) {
    const emailItem = document.getElementById(`email-item-${emailId}`);
    if (emailItem) {
        // Show success message briefly
        const successMsg = document.createElement('div');
        successMsg.className = 'floating-success';
        successMsg.textContent = message;
        emailItem.appendChild(successMsg);
        
        // Fade out and remove
        emailItem.style.transition = 'opacity 0.5s, transform 0.5s';
        emailItem.style.opacity = '0';
        emailItem.style.transform = 'translateX(-20px)';
        
        setTimeout(() => {
            emailItem.remove();
        }, 500);
    }
}

async function deleteEmail(emailId) {
    if (!confirm('⚠️ Delete this email from the database? This will remove all metadata but NOT delete from IMAP.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/emails/${emailId}`, {
            method: 'DELETE',
            credentials: 'include'
        });
        
        console.log('Delete response:', response.status);
        
        if (response.ok) {
            removeEmailFromList(emailId, '🗑️ Email deleted from database');
        } else {
            const errorText = await response.text();
            alert(`Failed to delete: ${errorText}`);
        }
    } catch (error) {
        console.error('Error deleting email:', error);
        alert('Error: ' + error.message);
    }
}

async function loadFolderHistory(emailId) {
    const container = document.getElementById(`folder-history-${emailId}`);
    if (!container) return;
    
    // Show loading state
    container.innerHTML = '<div style="color: #666; padding: 10px;">⏳ Loading location history...</div>';
    
    try {
        // Try to get location_history from email detail endpoint first
        const emailResponse = await fetch(`${API_BASE}/api/emails/${emailId}`, FETCH_OPTIONS);
        if (emailResponse.ok) {
            const emailData = await emailResponse.json();
            if (emailData.location_history && emailData.location_history.length > 0) {
                renderLocationHistory(emailData.location_history, container);
                return;
            }
        }
        
        // Fallback to folder-history endpoint
        const response = await fetch(`${API_BASE}/api/emails/${emailId}/folder-history`, FETCH_OPTIONS);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.history || data.history.length === 0) {
            container.innerHTML = '<div style="color: #666; padding: 10px;">📭 No location movements recorded</div>';
            return;
        }
        
        // Render history timeline (legacy format)
        const historyHtml = data.history.map((move, index) => {
            const date = new Date(move.moved_at);
            const dateStr = date.toLocaleString();
            const isFirst = index === data.history.length - 1;
            
            // Color code by moved_by
            const colorMap = {
                'user': '#3b82f6',
                'rule': '#10b981',
                'ai': '#8b5cf6',
                'system': '#64748b'
            };
            const color = colorMap[move.moved_by] || '#64748b';
            
            return `
                <div style="display: flex; gap: 10px; padding: 8px; background: #f8fafc; border-left: 3px solid ${color}; margin-bottom: 6px; border-radius: 4px;">
                    <div style="flex-shrink: 0; width: 140px; font-size: 0.85rem; color: #666;">
                        ${dateStr}
                    </div>
                    <div style="flex: 1;">
                        ${isFirst ? '📥' : '📂'} 
                        ${move.from_folder ? `<span style="text-decoration: line-through; color: #999;">${escapeHtml(move.from_folder)}</span> → ` : ''}
                        <strong>${escapeHtml(move.to_folder)}</strong>
                        <br>
                        <span style="font-size: 0.8rem; color: #666;">
                            by <strong style="color: ${color};">${move.moved_by}</strong>
                            ${move.move_reason ? ` - ${escapeHtml(move.move_reason)}` : ''}
                            ${move.time_in_previous_folder_seconds ? ` (${formatDuration(move.time_in_previous_folder_seconds)})` : ''}
                        </span>
                    </div>
                </div>
            `;
        }).reverse().join('');
        
        container.innerHTML = historyHtml;
    } catch (error) {
        console.error('Error loading location history:', error);
        container.innerHTML = `<div style="color: #ef4444; padding: 10px;">❌ Error loading location history: ${escapeHtml(error.message)}</div>`;
    }
}

function renderLocationHistory(history, container) {
    if (!history || history.length === 0) {
        container.innerHTML = '<div style="color: #666; padding: 10px;">📭 No location movements recorded</div>';
        return;
    }
    
    const historyHtml = history.map((move, index) => {
        const date = new Date(move.moved_at);
        const dateStr = date.toLocaleString();
        const isFirst = index === history.length - 1;
        const isCrossAccount = move.is_cross_account;
        
        // Color code by moved_by
        const colorMap = {
            'user': '#3b82f6',
            'ui': '#3b82f6',
            'rule': '#10b981',
            'cross_account_rule': '#10b981',
            'ai': '#8b5cf6',
            'system': '#64748b'
        };
        const color = colorMap[move.moved_by] || '#64748b';
        
        return `
            <div style="display: flex; gap: 10px; padding: 8px; background: #f8fafc; border-left: 3px solid ${color}; margin-bottom: 6px; border-radius: 4px;">
                <div style="flex-shrink: 0; width: 140px; font-size: 0.85rem; color: #666;">
                    ${dateStr}
                </div>
                <div style="flex: 1;">
                    ${isFirst ? '📥' : '📂'} 
                    ${move.from_account && move.from_account !== move.to_account ? 
                        `<span style="background: #fef3c7; padding: 2px 6px; border-radius: 3px; font-size: 0.8rem;">${escapeHtml(move.from_account)}</span>` : ''}
                    ${move.from_folder ? `<span style="text-decoration: line-through; color: #999;">${escapeHtml(move.from_folder)}</span> → ` : ''}
                    ${move.to_account && move.to_account !== move.from_account ? 
                        `<span style="background: #d1fae5; padding: 2px 6px; border-radius: 3px; font-size: 0.8rem;">${escapeHtml(move.to_account)}</span>` : ''}
                    <strong>${escapeHtml(move.to_folder)}</strong>
                    ${isCrossAccount ? ' <span style="color: #f59e0b;">🌐</span>' : ''}
                    <br>
                    <span style="font-size: 0.8rem; color: #666;">
                        by <strong style="color: ${color};">${move.moved_by}</strong>
                        ${move.move_reason ? ` - ${escapeHtml(move.move_reason)}` : ''}
                    </span>
                </div>
            </div>
        `;
    }).reverse().join('');
    
    container.innerHTML = historyHtml;
}

async function moveEmailToAccount(emailId) {
    const accountSelect = document.getElementById(`move-account-select-${emailId}`);
    const folderInput = document.getElementById(`move-folder-input-${emailId}`);
    
    if (!accountSelect || !accountSelect.value) {
        alert('Please select a target account');
        return;
    }
    
    const targetAccount = accountSelect.value;
    const targetFolder = folderInput.value.trim() || 'INBOX';
    
    if (!confirm(`Move email to ${targetAccount}:${targetFolder}?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/emails/${emailId}/move-to-account`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                target_account: targetAccount,
                target_folder: targetFolder
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            alert(`✅ ${data.message || 'Email moved successfully'}`);
            // Reload location history
            loadFolderHistory(emailId);
            // Reload email details to update account_id
            const emailItem = document.getElementById(`email-item-${emailId}`);
            if (emailItem) {
                // Trigger reload of email list if needed
                if (currentTab === 'browse') {
                    loadCategories();
                }
            }
        } else {
            const errorData = await response.json();
            alert(`❌ Failed to move email: ${errorData.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Error moving email:', error);
        alert(`❌ Error: ${error.message}`);
    }
}

function formatDuration(seconds) {
    if (seconds < 60) return `${seconds}s in prev`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m in prev`;
    if (seconds < 86400) return `${Math.round(seconds / 3600)}h in prev`;
    return `${Math.round(seconds / 86400)}d in prev`;
}

// ============================================================================
// Mail.app Integration - Open Reply with Prefilled Text
// ============================================================================

async function openReplyInMail(emailId, optionIndex) {
    const buttonId = `reply-btn-${emailId}`;
    const button = document.getElementById(buttonId);
    
    if (!button) {
        console.error('Reply button not found:', buttonId);
        return;
    }
    
    // Save original button state
    const originalText = button.innerHTML;
    const originalDisabled = button.disabled;
    
    // Hide dropdown if shown
    hideReplyDropdown(emailId);
    
    try {
        // Update button to show loading state
        button.disabled = true;
        button.innerHTML = '🔄 Opening...';
        
        // Fetch specific email to get answer options
        const emailResponse = await fetch(`${API_BASE}/api/emails/${emailId}`, FETCH_OPTIONS);
        
        if (!emailResponse.ok) {
            throw new Error('Failed to fetch email metadata');
        }
        
        const email = await emailResponse.json();
        const metadata = email.email_metadata || {};
        const answerOptions = metadata.answer_options || [];
        
        // Debug logging
        console.log('[DEBUG] Email data:', email);
        console.log('[DEBUG] Metadata:', metadata);
        console.log('[DEBUG] Answer options:', answerOptions);
        console.log('[DEBUG] Option index:', optionIndex);
        
        let replyText;
        
        if (optionIndex === 'all') {
            // Include all options
            replyText = answerOptions.map((opt, i) => 
                `[Option ${i+1}: ${opt.tone.toUpperCase()}]\n${opt.text}\n\n${'-'.repeat(60)}\n\n`
            ).join('');
        } else if (optionIndex !== null && optionIndex >= 0 && optionIndex < answerOptions.length) {
            // Use specific option
            replyText = answerOptions[optionIndex].text;
        } else {
            // Fallback to Lorem Ipsum if no options
            replyText = `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Best regards,`;
        }
        
        // Call API to open reply in Mail.app
        const response = await fetch(`${API_BASE}/api/emails/${emailId}/open-reply`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                reply_text: replyText
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to open reply');
        }
        
        const result = await response.json();
        console.log('✓ Reply opened in Mail.app:', result);
        
        // Show success state
        button.innerHTML = '✓ Opened!';
        button.style.background = '#dcfce7';
        button.style.borderColor = '#86efac';
        
        // Reset button after 2 seconds
        setTimeout(() => {
            button.innerHTML = originalText;
            button.disabled = originalDisabled;
            button.style.background = '';
            button.style.borderColor = '';
        }, 2000);
        
    } catch (error) {
        console.error('Error opening reply in Mail.app:', error);
        
        // Show error state
        button.innerHTML = '❌ Failed';
        button.style.background = '#fee2e2';
        button.style.borderColor = '#fca5a5';
        
        // Show detailed error to user
        const errorMessage = error.message || 'Unknown error';
        let helpText = '';
        
        if (errorMessage.includes('requires macOS') || errorMessage.includes('running on Linux')) {
            helpText = '\n\n💡 This feature requires running the web UI locally on your Mac.\n\n' +
                      'To use this feature:\n' +
                      '1. Run the web UI locally: cd web-ui && python3.11 app.py\n' +
                      '2. Access: http://localhost:8080\n\n' +
                      'All other features (search, browse, etc.) work on any server!';
        } else if (errorMessage.includes('not found in Mail.app') || errorMessage.includes('not be downloaded yet')) {
            helpText = '\n\n💡 The email isn\'t in Mail.app yet.\n\n' +
                      'Try this:\n' +
                      '1. Click "📬 Open in Mail" first to verify the email exists\n' +
                      '2. If it opens, the email is there - try Reply again\n' +
                      '3. If it doesn\'t open, wait for Mail.app to sync\n' +
                      '4. Check that Mail.app is not paused/offline';
        } else if (errorMessage.includes('timeout')) {
            helpText = '\n\nMail.app may not be responding. Try:\n• Restarting Mail.app\n• Checking if Mail.app is frozen';
        } else if (errorMessage.includes('message_id')) {
            helpText = '\n\nThis email doesn\'t have a valid Message-ID.';
        }
        
        alert(`Failed to open reply in Mail.app\n\nError: ${errorMessage}${helpText}`);
        
        // Reset button after 3 seconds
        setTimeout(() => {
            button.innerHTML = originalText;
            button.disabled = originalDisabled;
            button.style.background = '';
            button.style.borderColor = '';
        }, 3000);
    }
}

