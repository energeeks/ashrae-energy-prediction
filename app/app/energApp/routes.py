from flask import render_template, Blueprint, request
from flask_login import login_required, current_user
from .forms import BuildingForm
from .models import db, Building
from .weather import get_forecast, parse_request


main_bp = Blueprint('main_bp', __name__,
                    template_folder='templates',
                    static_folder='static')




@main_bp.route('/')
@login_required
def index():
    forecast = get_forecast(48, 11)
    forecast = parse_request(forecast)
    return render_template('index.html',
                           forecast=forecast.to_html())


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
