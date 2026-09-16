Humans struggle to understand compounding effects. In this task, you'll code an
interactive CLI simulation: (11_ratchet_simulator.py) to depict the timeline of
the intervention showcasing the feelings as well as the technical metrics that
will be logged during the treatment. The script should be interactive: it begins
by asking the user to input their starting age. If the user is too old (e.g., 90
or 100+), the therapy is impossible with current technology due to frailty, and
the script should recommend cryopreservation and exit. If the age is viable, it
performs the therapy, shows the result (both chronological and biological age),
and then prompts the user to come back in X years. The command line must
explicitly ask the user to agree to the therapy again each cycle. The narrative
arc: the first therapy is extremely physically challenging with a long recovery
time because the patient is older. However, because the therapy successfully
lowers their biological age, subsequent therapies become much easier to recover
from. The user should experience a joyful realization as recovery times
decrease. The simulation should end in victory once their biological age
reaches 20.
