let mymap = L.map('mapid').setView([48.137154, 11.576124], 13);

L.control.scale().addTo(mymap);

L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(mymap);

let searchControl = new L.esri.Controls.Geosearch().addTo(mymap);

let results = new L.LayerGroup().addTo(mymap);

searchControl.on('results', function(data){
    results.clearLayers();
    results.addLayer(L.marker(data.results[0].latlng, {draggable:'true'}));

    document.getElementById('latitude').value = data.results[0].latlng.lat;
    document.getElementById('longitude').value = data.results[0].latlng.lng;
    document.getElementById('name').value = data.results[0].address;
});
