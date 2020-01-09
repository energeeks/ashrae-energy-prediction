from flask import render_template, Blueprint
from flask_login import login_required

main_bp = Blueprint('main_bp', __name__,
                    template_folder='templates',
                    static_folder='static')


@main_bp.route('/')
@login_required
def index():
    return render_template('index.html')


@main_bp.route('/something')
@login_required
def do_something():
    return render_template('something.html')
