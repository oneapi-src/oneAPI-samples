"use strict";

const state = {
    data: [],
    search: [],
    pageSize: 15,
    totalPages: +Infinity,
    pageIndex: 1,
    paginatedData: [],
    checkedFilters: [],
    checkBoxMap: {
        getStart: /.*getting\sstarted.*/ig,
        tutorial: /.*tutorial.*/ig,
        conFunc: /.*concepts\sand\sfunctionality.*/ig,
        refDesEnd2End: /.*reference\sdesigns\sand\send\sto\send.*/ig,
        codeOpt: /.*code\soptimization.*/ig,
        cpp: /.*cpp.*/ig,
        fortran: /.*fortran.*/ig,
        python: /.*python.*/ig,
        cpu: /.*cpu.*/ig,
        gpu: /.*gpu.*/ig,
        fpga: /.*fpga.*/ig
    }
}

HTMLElement.prototype.onEvent = function (eventType, callBack, useCapture) { // est eventListener for all HTML Objs.
    this.addEventListener(eventType, callBack, useCapture);
    if (!this.myListeners) {
        this.myListeners = []; // create empty arr to hold event type, callbacks, captures in DOM
    }
    this.myListeners.push({ eType: eventType, callBack: callBack }); // populate same arr
    return this;
};

HTMLElement.prototype.removeListeners = function () {
    if (this.myListeners) { // If myListeners arr exists, run for loop over all eventListeners
        for (let i = 0; i < this.myListeners.length; i++) {
            this.removeEventListener(this.myListeners[i].eType, this.myListeners[i].callBack);
        }
        delete this.myListeners; // clear all eventListeners in arr before next func inits state
    }
};

document.addEventListener('DOMContentLoaded', function () {
    console.log('document is ready.');
    fetchData().then((data) => {
        renderUi(data.paginatedData);
        showTotalRecords();
        attachSearch();
        filterPaginatedData();
        // externalLinksOpenNewTab();
        lastUpdated();
    })
});


function reloadData() {
    const reload = new Event('DOMContentLoaded');
    document.dispatchEvent(reload)
    console.log(reload)
    externalLinksOpenNewTab();
}

function lastUpdated() {
    let updated = document.lastModified;
    document.getElementById("lastmodified").innerHTML = updated;
}

function fetchData() {
    return fetch('_static/sample_db_prd.json')
        .then(res => res.json())
        .then(data => { /* initialize state */
            state.data = [...data];
            state.search = [...data];
            state.pageSize = 15;
            state.pageIndex = 1;
            state.checkedFilters = [];
            state.totalPages = Math.ceil(data.length / state.pageSize);
            state.paginatedData = [...state.data].slice(0, state.pageSize);
            return { ...state };
        });
}

function printPaginatorState() {
    const paginatorStateEl = document.getElementById("paginator-state");
    paginatorStateEl.innerHTML = `Page ${state.pageIndex} / ${state.totalPages}`;
}

function firstPage() {
    state.paginatedData = [...state.search].slice(0, state.pageSize);
    state.pageIndex = 1;
    destroyView();
    renderUi(state.paginatedData);
}

function nextPage() {
    const nextIndex = state.pageIndex + 1;
    if (nextIndex <= state.totalPages) {
        const nextSliceStart = state.pageIndex * state.pageSize;
        state.paginatedData = [...state.search].slice(nextSliceStart, nextSliceStart + state.pageSize);
        state.pageIndex = nextIndex;
        destroyView();
        renderUi(state.paginatedData);
    }
}

function previousPage() {
    const nextIndex = state.pageIndex - 1;
    if (nextIndex >= 1) {
        const nextSliceStart = nextIndex * state.pageSize;
        state.paginatedData = [...state.search].slice(nextSliceStart - state.pageSize, nextSliceStart);
        state.pageIndex = nextIndex;
        destroyView();
        renderUi(state.paginatedData);
    }
}

function lastPage() {
    state.paginatedData = [...state.search].slice(state.search.length - state.pageSize, state.search.length);
    state.pageIndex = state.totalPages;
    destroyView();
    renderUi(state.paginatedData);
}

function resetFiltersGoHome(){
    resetFilters();
    destroyView();
    reloadData();
}

// function closeCollapsibleFilters(){
// //
// }


function resetFilters() {
    const checkboxes = document.querySelectorAll('input[name=data-filter]');
    console.log("NodeList:", checkboxes);
    checkboxes.forEach(c => {
        if (c.checked == true) {
            c.checked = false;
            const change = new Event('change');
            c.dispatchEvent(change);
        } else {
            console.log(c.checked);
        }
    });
}

function resetSearcher() {
    const input = document.getElementById('textInput');
    input.value = "";
    destroyView();
    resetFilters();
    reloadData();
    // input.focus();
}

function showTotalRecords() {
    const records = document.getElementById("total-records");
    records.innerHTML = `${[...state.data].length} total samples`;
}

function renderUi(data) {
    const filtered = state.checkedFilters.length === 0 ? data : data.filter(item => {
        const languages = item.languages.reduce((acc, value) => {
            const keys = Object.keys(value);
            return [...acc, ...keys];
        }, []);
        const filters = [...state.checkedFilters]
        const conditions = filters.map((checkboxValue) => {
            const regex = state.checkBoxMap[checkboxValue]
            const condition = regex.test(item.expertise) ||
                regex.test(languages.join(',')) ||
                regex.test(item.targetDevice.join(','));

            console.log(typeof (filters))
            console.log(filters)
            return condition
        })
        return conditions.reduce((acc, condition) => {
            return acc && condition
        }, true)

    })
    noResultsSwitch(filtered)
    qtyFilteredResults(filtered)
    renderCards(filtered)
    printPaginatorState()
    externalLinksOpenNewTab()
}

function qtyFilteredResults(filtered) {
    const qtyResults = filtered.length;
    const dataTotalLength = [...state.data].length;
    console.log("Total len", dataTotalLength)
    const results = document.getElementById("qty-show-results") || {};
    results.innerHTML = `${qtyResults} Results`;
    if (qtyResults === undefined || qtyResults == 15 || qtyResults == dataTotalLength) {
        results.style.display = "none";
    } else {
        results.style.display = "block";
    }
}

function attachSearch() {
    const searchInput = document.querySelector("[data-search]")
    const handler = e => {
        const value = e.target.value.replace(/\+/g, "\\+")
        const regex = new RegExp('.*' + saniStrings(value) + '.*', 'ig');
        search(regex);
    }
    searchInput.removeListeners()
    searchInput.onEvent("input", handler)
}

function noResultsSwitch(data) {
    const noResults = document.getElementById("hide")
    if (data === undefined || data.length == 0) {
        noResults.style.display = "block";
    } else {
        noResults.style.display = "none";
    }
}

function saniStrings(str) {
    /* remove reg/trademarks and all empty spaces  */
    const regNoSpecial = /[\u00ae\u2122\u2120*]/g;
    const result = str.replace(regNoSpecial, "");
    return result.trim().replace(/\s+/g, " ");
}


function search(regex) {
    const data = state.data.filter(item => {
        const languages = item.languages.reduce((acc, value) => {
            const keys = Object.keys(value);
            return [...acc, ...keys];
        }, []);
        let result =
            regex.test(saniStrings(item.name)) ||
            regex.test(saniStrings(item.description)) ||
            regex.test(languages.join(',')) ||
            regex.test(item.targetDevice.join(',')) ||
            regex.test(item.expertise);
        return result
    });
    noResultsSwitch(data)
    state.search = [...data];
    state.totalPages = Math.ceil(data.length / state.pageSize);
    state.paginatedData = [...data].slice(0, state.pageSize);
    destroyView();
    renderUi(data);
}

function onlyAlphaNumInput(e) {
    let regex1 = new RegExp("^[a-z-A-Z0-9\-\r|\n ]+|[\u00ae\u2122\u2120]+|[+]+$")
    let str = String.fromCharCode(!e.fromCharCode ? e.which : e.fromCharCode);
    saniStrings(str)
    const errormsg = document.getElementById("errormsg");
    if (regex1.test(str)) {
        errormsg.style.display = "none";
        return true;
    } else {
        errormsg.innerHTML = "Enter only letters, numbers, or plus sign.";
        errormsg.classList.toggle("showerror");
        setTimeout(() => {
            errormsg.classList.remove("showerror");
        }, 7000);
    }
    e.preventDefault();
}

function checkboxFilterHandler(checked, checkboxValue) {
    // where checkboxValue is arr
    if (checked) {
        state.checkedFilters.push(checkboxValue)
        console.log("state.checkedFilters:",state.checkedFilters)
    } else {
        const found = state.checkedFilters.findIndex(item => {
            return item === checkboxValue
        })
        console.log("found?", found)
        if (found !== -1) {
            state.checkedFilters.splice(found, 1)
        }
        // console.log(`Checkbox not checked.`);
    }
    destroyView();
    if (state.checkedFilters.length >= 0) {
        // changed above from > 0 to > -1 // 25/10/2023
        state.pageIndex = 1;
        state.pageSize = +Infinity;
        state.totalPages = 1;
        renderUi([...state.data]);
    } else {

        reloadData();
    }
}

function filterPaginatedData() {
    const checkboxes = document.querySelectorAll('input[name=data-filter]');
    console.log(checkboxes)

    for (let checkbox of checkboxes) {
        const handler = e => {
            const checked = e.target.checked;
            // console.log("Checked is T/F?",checked)
            const checkedParent = e.target.parentElement.parentElement
            const checkBoxGroup = checkedParent.querySelectorAll('input[name=data-filter]')
            console.log("checkBoxGroup", checkBoxGroup)
            // 
            checkBoxGroup.forEach(item => {

                if (item.value !== e.target.value && item.checked == true) {
                    item.checked = false
                    const uncheckRegex = state.checkBoxMap[item.value]
                    console.log("Unchecked regex...?", uncheckRegex)
                    checkboxFilterHandler(item.checked, item.value)
                }
            })
            checkboxFilterHandler(checked, e.target.value)
        }
        checkbox.removeListeners()       
        checkbox.onEvent('change', handler)

    }
}

function destroyView() {
    const toolkitCardContainer = document.querySelector("[data-toolkit-cards-container]");
    const cards = [...toolkitCardContainer.children];
    for (let i = 0, max = cards.length; i < max; i += 1) {
        const card = cards[i];
        toolkitCardContainer.removeChild(card);
    }
}

function renderCards(data) {
    const toolkitCardTemplate = document.querySelector("[data-toolkit-template]")
    const toolkitCardContainer = document.querySelector("[data-toolkit-cards-container]")

    data.map(toolkit => {
        const card = toolkitCardTemplate.content.cloneNode(true).children[0]
        const title = card.querySelector("[data-header]")
        const body = card.querySelector("[data-body]")
        const expertise = card.querySelector("[data-expertise]")
        const languages = card.querySelector("[data-languages]")
        const targetDevice = card.querySelector("[data-targetDevice]")
        const url = card.querySelector("[url]")

        title.textContent = toolkit.name
        body.textContent = toolkit.description
        expertise.textContent = toolkit.expertise
        languages.innerHTML = Object.keys(toolkit.languages[0])
        targetDevice.innerHTML = '<p class="badges badge3">' + toolkit.targetDevice.join('</p><p class="badges badge3" >') + '</p>'
        url.innerHTML = '<a href="' + toolkit.url + '"></a>'

        toolkitCardContainer.append(card)

        return {
            name: toolkit.name,
            description: toolkit.description,
            expertise: toolkit.expertise,
            languages: toolkit.languages,
            targetDevice: toolkit.targetDevice,
            url: toolkit.url,
            element: card
        }
    })
}

function externalLinksOpenNewTab() {

    const hyperlinks = document.querySelectorAll("a[href^='https://'], a[href^='http://']");
    const host = window.location.hostname;
    const internalLink = link => new URL(link).hostname === host

    hyperlinks.forEach(link => {
        if (internalLink(link)) return        
        link.setAttribute("target", "_blank")
        link.setAttribute("rel", "noopener")
    })

};
