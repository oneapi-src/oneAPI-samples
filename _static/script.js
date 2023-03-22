"use strict";
const state = {
    data: []
}

function fetchData() {
    return fetch('_static/sample_db_prd.json')
        .then(res => res.json())
        .then(data => {
            state.data = [...data]; /* default array - no handling */
            return data   
        })
};

document.addEventListener('DOMContentLoaded', function () {
    console.log('document is ready.');
    fetchData().then((data) => { /* */
        renderUi(data); /* */
        console.log(data)
    });
});

function renderUi(data) {
    renderCards(data)
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
        targetDevice.innerHTML = '<p class="expertise badge3">' + toolkit.targetDevice.join('</p><p class="expertise badge3" >') + '</p>'
        url.innerHTML = '<a href="' + toolkit.url + '">Hyperlink</a>'

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
