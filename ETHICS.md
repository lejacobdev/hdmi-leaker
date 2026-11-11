# Ethics & acceptable use

This project exists to **teach** the signal theory behind video-cable
electromagnetic leakage and to help people **defend** their own systems. Please
keep it in that lane.

## Acceptable

- Learning how compromising emanations arise and how reconstruction works.
- Simulating leakage from images/screenshots you supply.
- Reconstructing IQ you recorded from **your own** equipment, or equipment you
  are **explicitly, lawfully authorized** to test (e.g. a sanctioned emissions
  assessment, a red-team engagement with written scope, a CTF, academic research
  on your own lab gear).
- Evaluating and tuning countermeasures (shielding, TEMPEST-safe fonts, mode
  choices).

## Not acceptable

- Capturing or reconstructing emanations from anyone else's equipment without
  authorization. In most jurisdictions this is a criminal offence under wiretap
  / unauthorized-interception law, independent of what you do with the result.
- Using this toolkit to surveil a person, household, or organization.

## Why the live-capture path is gated

`capture.py` performs no scanning, targeting, or device identification, and it
refuses to run until the caller passes `--i-am-authorized`, acknowledging they
own or are authorized to test the source. That gate is a deliberate speed bump,
not a legal shield — the responsibility to be authorized is yours.

If you are assessing real equipment, document your authorization before you
start, stay within scope, and follow your local law and your organization's
rules.
