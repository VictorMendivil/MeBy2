#resources/app 

- Integration Restricted is created to keep the ETL mechanism for ECOM (restricted) data in a separate system. 
- This system have two server instances; one for Informatica Powercenter, and one database server for the repository of Powercenter. 
- The purpose is to house the ETL jobs for [[Databronn system]], [[Datatorget system]] and Boygen. It will then be an interface between the different source systems and the databases for the mentioned Datawarehouse's. 