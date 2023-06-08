from django.urls import path
from . import views

urlpatterns = [
    path("",views.home),
    path("home",views.home),
    path("prediction",views.prediction),
    path("team",views.team),
    path("login",views.login),
    path("register",views.register),
    path("logout",views.logout_request)

]