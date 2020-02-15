// Adds listener to all checkboxes that when changed the graph will be updated.
let building_meter = document.getElementsByClassName("meter-types");
for (let i = 0; i < building_meter.length; i++) {
  building_meter[i].addEventListener("change", function() {
        let x = (this.getAttribute("data-index"));
        $.ajax({
            url: "/plot",
            type: "GET",
            contentType: "application/json;charset=UTF-8",
            data: {
                "building": this.getAttribute("data-building"),
                "m0": $("#b"+x+"-m0").is(':checked') ? 1 : 0,
                "m1": $("#b"+x+"-m1").is(':checked') ? 1 : 0,
                "m2": $("#b"+x+"-m2").is(':checked') ? 1 : 0,
                "m3": $("#b"+x+"-m3").is(':checked') ? 1 : 0,
                "at": $("#b"+x+"-at").is(':checked') ? 1 : 0
            },
            dataType:"json",
            success: function (plot) {
                Plotly.newPlot("building"+x, plot.data, plot.layout);
            }
        });
});
}

