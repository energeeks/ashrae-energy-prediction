from . import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


class User(UserMixin, db.Model):
    """Model for user accounts."""

    __tablename__ = 'user'

    id = db.Column(db.Integer,
                   primary_key=True)
    name = db.Column(db.String,
                     nullable=False,
                     unique=False)
    password = db.Column(db.String(200),
                         primary_key=False,
                         unique=False,
                         nullable=False)
    buildings = db.relationship('Building', backref='user', lazy=True)

    def set_password(self, password):
        """Create hashed password."""
        self.password = generate_password_hash(password, method='sha256')

    def check_password(self, password):
        """Check hashed password."""
        return check_password_hash(self.password, password)

    def __repr__(self):
        return '<User {}>'.format(self.name)


class Building(db.Model):
    """Model for a building that the users enters in their profile"""

    __tablename__ = 'building'

    id = db.Column(db.Integer,
                   primary_key=True)
    name = db.Column(db.String,
                     nullable=False,
                     unique=False)
    primary_use = db.Column(db.String(120),
                            nullable=False)
    square_feet = db.Column(db.Integer,
                            nullable=False)
    year_built = db.Column(db.Integer,
                           nullable=True)
    floorcount = db.Column(db.Integer,
                           nullable=True)
    latitude = db.Column(db.Float,
                             nullable=False)
    longitude = db.Column(db.Float,
                              nullable=False)
    user_id = db.Column(db.Integer,
                        db.ForeignKey('user.id'),
                        nullable=False)

    def __repr__(self):
        return '<Building {}>'.format(self.name)
