import pandas as pd
from flask import render_template, Blueprint, request, current_app
from flask_login import login_required, current_user
from .forms import BuildingForm
from .models import db, Building
from .weather import get_forecast, parse_request
from .predict import predict_energy_consumption
from .graph import create_plot

main_bp = Blueprint('main_bp', __name__,
                    template_folder='templates',
                    static_folder='static')


@main_bp.route('/')
@login_required
def index():
    building_query = Building.query.filter_by(user_id=current_user.name)
    buildings = pd.read_sql(building_query.statement,
                            building_query.session.bind)
    forecasts = [get_forecast(r["latitude"], r["longitude"]) for i, r in buildings.iterrows()]
    forecasts = [parse_request(f) for f in forecasts]

    predictions = predict_energy_consumption(model=current_app.config["MODEL"],
                                             buildings=buildings,
                                             forecasts=forecasts)
    plot = create_plot(1, 1, 1, 1)

    return render_template('index.html',
                           forecasts=forecasts,
                           plot=plot,
                           model=predictions)


@main_bp.route('/plot', methods=['GET', 'POST'])
def change_meters():
    meter0 = int(request.args["m0"])
    meter1 = int(request.args["m1"])
    meter2 = int(request.args["m2"])
    meter3 = int(request.args["m3"])
    return create_plot(meter0, meter1, meter2, meter3)


@main_bp.route('/something')
@login_required
def do_something():
    return render_template('something.html')


@main_bp.route('/buildings', methods=['GET', 'POST'])
@login_required
def buildings_page():
    building_form = BuildingForm(request.form)
    if request.method == 'POST':
        if building_form.validate():
            building = Building(name=request.form.get('name'),
                                primary_use=request.form.get('primary_use'),
                                square_feet=request.form.get('square_feet'),
                                year_built=request.form.get('year_built'),
                                floorcount=request.form.get('floorcount'),
                                latitude=request.form.get('latitude'),
                                longitude=request.form.get('longitude'),
                                user_id=current_user.name)
            db.session.add(building)
            db.session.commit()

    buildings = Building.query.filter_by(user_id=current_user.name).all()
    return render_template('building.html', buildings=buildings, form=BuildingForm())
