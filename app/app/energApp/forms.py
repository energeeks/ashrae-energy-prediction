from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, validators, SubmitField, \
    IntegerField, FloatField, HiddenField, SelectField
from wtforms.validators import ValidationError, DataRequired, \
    EqualTo, Length


class SignupForm(FlaskForm):
    """
    User interface for signing up at the app.
    """
    name = StringField('Name',
                       validators=[DataRequired(message=('Please enter a nickname.'))])
    password = PasswordField('Password',
                             validators=[DataRequired(message='Please enter a password.'),
                                         Length(min=3, message=('Please select a stronger password.')),
                                         EqualTo('confirm', message='Passwords must match')])
    confirm = PasswordField('Confirm Your Password', )
    submit = SubmitField('Register')


class LoginForm(FlaskForm):
    """
    User interface for signing up at the app.
    """
    name = StringField('Name', validators=[DataRequired('Please enter a valid name.')])
    password = PasswordField('Password', validators=[DataRequired('Please enter your password.')])
    submit = SubmitField('Log In')


class BuildingForm(FlaskForm):
    """
    Interface for adding a building to a user account.
    """
    name = StringField('Name')
    fields = [
        ('Education', 'Education'),
        ('Entertainment/public assembly', 'Entertainment/public assembly'),
        ('Food sales and service', 'Food sales and service'),
        ('Healthcare', 'Healthcare'),
        ('Lodging/residential', 'Lodging/residential'),
        ('Manufacturing/industrial', 'Manufacturing/industrial'),
        ('Office', 'Office'),
        ('Other', 'Other'),
        ('Parking', 'Parking'),
        ('Public services', 'Public services'),
        ('Religious worship', 'Religious worship'),
        ('Retail', 'Retail'),
        ('Services', 'Services'),
        ('Technology/science', 'Technology/science'),
        ('Utility', 'Utility'),
        ('Warehouse/storage', 'Warehouse/storage')
        ]
    primary_use = SelectField('Primary Use', choices=fields)
    square_feet = IntegerField('Square Feet')
    year_built = IntegerField('Year Built')
    floorcount = IntegerField('Floors')
    latitude = HiddenField('Latitude')
    longitude = HiddenField('Longitude')
    submit = SubmitField('Save')
