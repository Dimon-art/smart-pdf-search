import pymupdf


SECTIONS = [
    ("1. Subject of the Agreement",
     "The Contractor agrees to provide software development services. "
     "The scope includes design, development, testing, and deployment of the software. "
     "The Client agrees to accept and pay for the services in accordance with this Agreement."),

    ("2. Payment Terms",
     "Payment shall be made within 30 calendar days from the date of signing the acceptance act. "
     "The total contract value is 50000 USD. Payment is made by bank transfer to the account specified in the invoice."),

    ("3. Penalties and Late Payment",
     "For violation of payment terms, the Client shall pay a penalty of 0.1 percent of the outstanding amount "
     "for each day of delay. The penalty shall not exceed 10 percent of the total contract value."),

    ("4. Warranty and Support",
     "The Contractor provides a warranty of 12 months from the date of acceptance. "
     "During the warranty period, all defects shall be fixed free of charge. "
     "Support is provided by email during business hours."),

    ("5. Confidentiality",
     "Both parties agree to keep confidential all information received during the performance of this Agreement. "
     "Confidential information shall not be disclosed to third parties without prior written consent."),

    ("6. Intellectual Property",
     "All intellectual property rights to the software developed under this Agreement "
     "shall belong to the Client after full payment. The Contractor retains the right to use "
     "generic knowledge and skills acquired during the project."),

    ("7. Force Majeure",
     "The parties shall be released from liability upon the occurrence of force majeure circumstances, "
     "including natural disasters, wars, strikes, and government actions. "
     "The affected party shall notify the other party within 14 days."),

    ("8. Termination",
     "Either party may terminate this Agreement with 30 days written notice. "
     "Upon termination, the Client shall pay for all services rendered up to the termination date."),

    ("9. Dispute Resolution",
     "All disputes shall be resolved through negotiations. "
     "If no agreement is reached within 30 days, the dispute shall be submitted to the arbitration court "
     "at the location of the defendant."),

    ("10. Personal Data",
     "The Contractor shall process personal data in accordance with applicable data protection laws. "
     "Personal data shall not be transferred to third parties except as required by law."),

    ("11. Delivery and Acceptance",
     "The Contractor shall deliver the software in electronic form via secure file transfer. "
     "The Client shall review and accept the deliverables within 10 business days."),

    ("12. Amendments",
     "Any amendments to this Agreement shall be made in writing and signed by both parties. "
     "Verbal agreements shall not be considered valid."),
]


def main():
    doc = pymupdf.open()
    for title, body in SECTIONS:
        page = doc.new_page()
        text = f"{title}\n\n{body}"
        page.insert_text((50, 50), text, fontsize=11)
    doc.save('../sample.pdf')
    doc.close()
    print(f'Создан sample.pdf с {len(SECTIONS)} страницами')


if __name__ == '__main__':
    main()
