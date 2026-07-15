# Production deployment plan

## Recommended AWS setup
- Use an Ubuntu EC2 instance (t3.micro for light traffic, t3.small if you expect more concurrent users).
- Allocate a static Elastic IP and map it to your Route 53 A record.
- Install Nginx, Certbot, Python 3.10+, and a virtual environment for the app.
- Run Streamlit as a systemd service and proxy traffic through Nginx.

## Cost expectations
- EC2: about $8–$16/month for a t3.micro or t3.small in most regions.
- EBS storage: roughly $1–$3/month for 20–30 GB.
- Route 53 hosted zone: typically $0.50/month plus record costs if you use a public hosted zone.
- Nginx/Certbot: free.
- Total expected spend: roughly $10–$20/month for the baseline setup, assuming light traffic and modest storage.

## Polite scraping strategy
- Keep concurrency at 1 request at a time.
- Use a minimum interval of 1.5 seconds between outbound requests.
- Cache responses aggressively in Streamlit so repeat page loads do not hit the provider again.
- Prefer local cached data for common views and only refresh when needed.

## Ubuntu deployment steps
1. Update the server and install dependencies.
2. Clone or copy the project to /home/ubuntu/mutual-fund-analysis.
3. Create a Python virtual environment and install requirements.
4. Configure Nginx with the provided nginx.conf template.
5. Obtain an SSL certificate with Certbot.
6. Create a systemd service file for the Streamlit app.
