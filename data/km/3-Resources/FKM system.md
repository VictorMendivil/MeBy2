#resources/app  

* ID: S00833
* Stands for Felles KundeMaster
* Stores data for mobile about products, accounts, subscriptions, customers, subscribed products and agreements and much more. The purpose is  continuous load  of product, account, subscription, customer, subscribed product and agreement data from [[FKM system]] to the [[CCDW system]] staging area.
* It's not running on a mainframe
* These databases are running in [[Sybase]]
* [[SAP Hana system]] has a replica of the data for performance reasons
* The target is to migrate to [[Azure Db]]
* Reference here: [FKM(S00833) - DWH - PRIMA Confluence](https://prima.corp.telenor.no/confluence/pages/viewpage.action?pageId=51091721)