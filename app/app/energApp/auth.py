from flask import redirect, render_template, flash, Blueprint, request, url_for
from flask_login import login_required, logout_user, current_user, login_user
from werkzeug.security import generate_password_hash
from .forms import LoginForm, SignupForm
from .models import db, User
from . import login_manager

auth_bp = Blueprint('auth_bp', __name__,
                    template_folder='templates',
                    static_folder='static')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login_page():
    """
    Interface for logging in a user.
    """
    if current_user.is_authenticated:
        return redirect(url_for('main_bp.index'))
    login_form = LoginForm(request.form)
    if request.method == 'POST':
        if login_form.validate():
            name = request.form.get('name')
            password = request.form.get('password')
            user = User.query.filter_by(name=name).first()
            if user:
                if user.check_password(password=password):
                    login_user(user)
                    next = request.args.get('next')
                    return redirect(next or url_for('main_bp.index'))
        flash('Invalid username or password')
        return redirect(url_for('auth_bp.login_page'))
    return render_template('/login.html', form=LoginForm())


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup_page():
    """
    Interface for signing up and therefore creating a user.
    """
    signup_form = SignupForm(request.form)
    if request.method == 'POST':
        if signup_form.validate():
            name = request.form.get('name')
            password = request.form.get('password')
            existing_user = User.query.filter_by(name=name).first()
            if existing_user is None:
                user = User(name=name,
                            password=generate_password_hash(password, method='sha256'))
                db.session.add(user)
                db.session.commit()
                login_user(user)
                return redirect(url_for('main_bp.index'))
            flash('A user already exists with that nickname.')
            return redirect(url_for('auth_bp.signup_page'))
    return render_template('/signup.html', form=SignupForm())


@auth_bp.route("/logout")
@login_required
def logout_page():
    """
    User log-out logic.
    """
    logout_user()
    return redirect(url_for('main_bp.index'))


@login_manager.user_loader
def load_user(user_id):
    """
    Check if user is logged-in on every page load.
    """
    if user_id is not None:
        return User.query.get(user_id)
    return None


@login_manager.unauthorized_handler
def unauthorized():
    """
    Redirect unauthorized users to Login page.
    """
    flash('You must be logged in to view that page.')
    return redirect(url_for('auth_bp.login_page'))