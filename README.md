# Passive Browser Detection in TLS Encrypted Traffic

Browser detection is crucial for enhancing and strengthening cybersecurity by defining access
controls, detecting suspicious activity, and most importantly provide visibility into encrypted
traffic. By implementing effective browser detection, enterprises can enhance their understanding
of encrypted traffic, enabling proactive protection against potential security risks. Indeed,
this visibility and insight into browsers also empowers them to safeguard their systems and
data, mitigating the vulnerabilities commonly exploited by attackers. Traditional techniques
to browser detection frequently depend on passive fingerprinting, which has major limitations
such as erroneous or outdated fingerprints. To address these constraints, we implemented
a machine learning-based solution that use classifiers trained on Transmission Control Protocol/
Internet Protocol (TCP/IP) headers and Transport Layer Security (TLS) messages to
provide greater precision and adaptability for cybersecurity applications. Our approach entails
collecting and preprocessing network traffic data in order to extract various TCP/IP, TLS, and
generic packet properties. Following that, we used feature selection to determine the most
relevant features, resulting in a balance between accuracy and training as well as testing time.
Then the model selection stage is carried out to identify the best model. The performance of
the classifier is assessed using various test sets containing traffic with known browser labels.
With an accuracy of 0.91%, our results show that the machine learning approach is effective
at passively detecting the browser in encrypted TLS traffic. This method has the potential
to increase network security and forensics efficacy by providing insight into the devices and
systems present on a network when communication is encrypted.