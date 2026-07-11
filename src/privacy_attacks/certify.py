"""Entry point so ``python -m privacy_attacks.certify`` runs the certification CLI."""

from privacy_attacks.certification.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
