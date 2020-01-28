import os
from lightgbm import Booster
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

db = SQLAlchemy()
login_manager = LoginManager()


def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'hJp3ZCMLRvChfj8XpuQv48jTNEC8WPIm'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://ener:geek@db/energeek_app'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    login_manager.init_app(app)
    Bootstrap(app)

    with app.app_context():
        from . import routes
        from . import auth
        from . import config
        app.register_blueprint(routes.main_bp)
        app.register_blueprint(auth.auth_bp)

        # Load weather API Key
        app.config['API_KEY'] = config.api_key

        # Load model
        app.config['MODEL'] = Booster(model_file="energApp/models/"
                                                + os.listdir("energApp/models/")[0])

        db.create_all()
        # db.drop_all()

    from .models import User, Building
    admin = Admin(app)
    admin.add_view(ModelView(User, db.session))
    admin.add_view(ModelView(Building, db.session))

    return app
