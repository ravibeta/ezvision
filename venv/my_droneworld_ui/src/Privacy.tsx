// src/pages/Privacy.tsx

import React from "react";

const Privacy: React.FC = () => {
  return (
    <main style={styles.container}>
      <h1 style={styles.heading}>Privacy Policy</h1>
      <p><strong>Effective Date:</strong> August 14, 2025</p>

      <section style={styles.section}>
        <h2>1. Information We Collect</h2>
        <ul>
          <li><strong>Personal Information:</strong> Name, email address, account credentials, and other identifiers you provide during registration.</li>
          <li><strong>Drone Video Content:</strong> Uploaded video files, metadata (e.g., GPS coordinates, timestamps), and analysis results.</li>
          <li><strong>Usage Data:</strong> IP address, browser type, device info, pages visited, and interactions with our platform.</li>
          <li><strong>Cookies and Tracking:</strong> We use cookies and similar technologies to enhance user experience and monitor performance.</li>
        </ul>
      </section>

      <section style={styles.section}>
        <h2>2. How We Use Your Information</h2>
        <ul>
          <li>Provide and improve our drone video analysis services</li>
          <li>Authenticate users and manage accounts</li>
          <li>Communicate updates, support, or promotional offers</li>
          <li>Ensure platform security and prevent misuse</li>
          <li>Comply with legal obligations</li>
        </ul>
      </section>

      <section style={styles.section}>
        <h2>3. Sharing and Disclosure</h2>
        <p>We do <strong>not</strong> sell your personal data. We may share your information with:</p>
        <ul>
          <li><strong>Service Providers:</strong> Vendors assisting with hosting, analytics, or support</li>
          <li><strong>Legal Authorities:</strong> When required by law or to protect rights</li>
          <li><strong>Business Transfers:</strong> In case of merger, acquisition, or asset sale</li>
        </ul>
      </section>

      <section style={styles.section}>
        <h2>4. Data Storage and Security</h2>
        <p>Your data is stored securely using encryption and access controls. We retain data only as long as needed for service delivery or legal compliance.</p>
      </section>

      <section style={styles.section}>
        <h2>5. Your Rights and Choices</h2>
        <p>You may have rights to:</p>
        <ul>
          <li>Access, correct, or delete your personal data</li>
          <li>Withdraw consent for data processing</li>
          <li>Object to certain uses of your data</li>
        </ul>
        <p>To exercise these rights, contact us at <a href="mailto:ravi@ezcloudiac.com">ravi@ezcloudiac.com</a>.</p>
      </section>

      <section style={styles.section}>
        <h2>6. Children's Privacy</h2>
        <p>Our platform is not intended for individuals under 13. We do not knowingly collect data from children.</p>
      </section>

      <section style={styles.section}>
        <h2>7. Changes to This Policy</h2>
        <p>We may update this Privacy Policy periodically. Changes will be posted here with a revised effective date.</p>
      </section>

      <section style={styles.section}>
        <h2>8. Contact Us</h2>
        <p>If you have questions or concerns, reach out to:</p>
        <p><strong>EZCloud IAC</strong><br />
          <br />
          Email: <a href="mailto:ravi@ezcloudiac.com">ravi@ezcloudiac.com</a>
        </p>
      </section>
    </main>
  );
};

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    padding: "2rem",
    maxWidth: "800px",
    margin: "0 auto",
    backgroundColor: "#fff",
    color: "#333",
    fontFamily: "Arial, sans-serif",
  },
  heading: {
    color: "#2c3e50",
    fontSize: "2rem",
    marginBottom: "1rem",
  },
  section: {
    marginBottom: "2rem",
  },
};

export default Privacy;
