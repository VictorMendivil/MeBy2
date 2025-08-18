#resources/telenor 

![[Pasted image 20250326132334.png]]
* Network: Tracks customer usage (e.g., calls, data, SMS) and generates records of what services were used, when, and how much.
* Order: Manages the setup and changes of services (e.g., new subscriptions, plan upgrades, add-ons) that need to be billed.
	* [[FKM system]]
	- [[MOOx system]]
- Mediation: Acts as a bridge between the network and billing system. It collects raw usage data (e.g., calls, data sessions, SMS logs) from network elements, processes and converts it into a standard format, and then feeds it into the billing system. Mediation ensures accuracy, filtering out errors and applying necessary adjustments before billing.
	- [[Event Pre-Processor (EPP) system]]
* Other billing sources
	* [[Cisco Jasper (TKS) system]]
	* [[Norwegian International Ship Register (NIS) system]]
	* [[Strex system]]
	* CPA. Content Provider Access
* Party: Manages customer and account-related information. This includes individuals or businesses using the services, their contracts, payment preferences, and relationships (e.g., a company with multiple employee subscriptions under one account). The party module ensures that billing is assigned to the right entity and that customer profiles are up to date.
	* [[Party Management system]]
* Revenue: Ensures that charges for services are correctly calculated and assigned to customers, including pricing, discounts, and promotions.
	* [[deFakto system]]
	- [[Marius system]]
	- [[Sinsen system]]
	- [[Geneva system]]
	- [[ARGO system]]
* Dunning, collections, accounting: Handles the invoicing process, payment collection, overdue reminders (dunning), and financial reporting to ensure accurate accounting.
	* [[Account Receivable Norway (ARN) app]]
* Document formatting, storage and distribution: Generates, formats, and stores invoices, bills, and other financial documents for customers and compliance purposes.
	* [[CCDW system]]
	- [[ISS system]]
	- [[Invoice hotel system]]
	- [[TRex system]]
