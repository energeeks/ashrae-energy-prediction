let mymap = L.map('mapid').setView([48.137154, 11.576124], 13);

L.control.scale().addTo(mymap);

L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(mymap);

let searchControl = new L.esri.Controls.Geosearch().addTo(mymap);

let results = new L.LayerGroup().addTo(mymap);

let marker = L.marker([48.137154, 11.576124], {draggable:'true'});
results.addLayer(marker);

searchControl.on('results', function(data){
    results.clearLayers();
    marker = L.marker(data.results[0].latlng, {draggable:'true'})
    results.addLayer(marker);

    document.getElementById('latitude').value = data.results[0].latlng.lat;
    document.getElementById('longitude').value = data.results[0].latlng.lng;
    document.getElementById('name').value = data.results[0].address;
});

marker.on('dragend', function(event) {
    let result = event.target.getLatLng();
    document.getElementById('latitude').value = result.lat;
    document.getElementById('longitude').value = result.lng;
});

$('#buildingModal').on('shown.bs.modal', function(){
    mymap.invalidateSize();
 });


let building_id = document.getElementsByClassName("building-id");
let current_building;
for (let i = 0; i < building_id.length; i++) {
  building_id[i].addEventListener("click", function() {
        current_building = (this.getAttribute("data-bid"));
});
}


let delete_building = document.getElementsByClassName("delete-building");

for (let i = 0; i < delete_building.length; i++) {
  delete_building[i].addEventListener("click", function() {
        $.ajax({
            url: "/delete_building",
            type: "GET",
            contentType: "application/json;charset=UTF-8",
            data: {
                "building": current_building
            },
            success: function () {
                window.location.href = "buildings";
            }
        });
});
}