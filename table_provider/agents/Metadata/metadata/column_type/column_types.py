from .measure_types import SubCategory

ColumnType = {
    "T2Dv2": [
        # Dim
        "http://dbpedia.org/ontology/address",
        "http://dbpedia.org/ontology/alias",
        "http://dbpedia.org/ontology/almaMater",
        "http://dbpedia.org/ontology/artist",
        "http://dbpedia.org/ontology/author",
        "http://dbpedia.org/ontology/award",
        "http://dbpedia.org/ontology/capital",
        "http://dbpedia.org/ontology/category",
        "http://dbpedia.org/ontology/child",
        "http://dbpedia.org/ontology/city",
        "http://dbpedia.org/ontology/class",
        "http://dbpedia.org/ontology/code",
        "http://dbpedia.org/ontology/commonName",
        "http://dbpedia.org/ontology/computingPlatform",
        "http://dbpedia.org/ontology/conservationStatus",
        "http://dbpedia.org/ontology/continent",
        "http://dbpedia.org/ontology/country",
        "http://dbpedia.org/ontology/currencyCode",
        "http://dbpedia.org/ontology/description",
        "http://dbpedia.org/ontology/developer",
        "http://dbpedia.org/ontology/diocese",
        "http://dbpedia.org/ontology/director",
        "http://dbpedia.org/ontology/doctoralAdvisor",
        "http://dbpedia.org/ontology/family",
        "http://dbpedia.org/ontology/fipsCode",
        "http://dbpedia.org/ontology/formerName",
        "http://dbpedia.org/ontology/founder",
        "http://dbpedia.org/ontology/frenchName",
        "http://dbpedia.org/ontology/genre",
        "http://dbpedia.org/ontology/genus",
        "http://dbpedia.org/ontology/governmentType",
        "http://dbpedia.org/ontology/headquarter",
        "http://dbpedia.org/ontology/iataAirlineCode",
        "http://dbpedia.org/ontology/iataLocationIdentifier",
        "http://dbpedia.org/ontology/industry",
        "http://dbpedia.org/ontology/iso31661Code",
        "http://dbpedia.org/ontology/isoCode",
        "http://dbpedia.org/ontology/knownFor",
        "http://dbpedia.org/ontology/language",
        "http://dbpedia.org/ontology/locatedInArea",
        "http://dbpedia.org/ontology/location",
        "http://dbpedia.org/ontology/manufacturer",
        "http://dbpedia.org/ontology/mayor",
        "http://dbpedia.org/ontology/mountainRange",
        "http://dbpedia.org/ontology/movement",
        "http://dbpedia.org/ontology/officialLanguage",
        "http://dbpedia.org/ontology/origin",
        "http://dbpedia.org/ontology/owner",
        "http://dbpedia.org/ontology/populationMetro",
        "http://dbpedia.org/ontology/portrayer",
        "http://dbpedia.org/ontology/programmeFormat",
        "http://dbpedia.org/ontology/publisher",
        "http://dbpedia.org/ontology/region",
        "http://dbpedia.org/ontology/religion",
        "http://dbpedia.org/ontology/spouse",
        "http://dbpedia.org/ontology/subdivision",
        "http://dbpedia.org/ontology/symbol",
        "http://dbpedia.org/ontology/synonym",
        "http://dbpedia.org/ontology/team",
        "http://dbpedia.org/ontology/topLevelDomain",
        "http://dbpedia.org/ontology/type",
        "http://dbpedia.org/ontology/usingCountry",
        "http://dbpedia.org/ontology/usk",
        "http://dbpedia.org/ontology/writer",
        "http://www.w3.org/2000/01/rdf-schema#label",

        # Both
        "http://dbpedia.org/ontology/activeYearsEndDate",
        "http://dbpedia.org/ontology/activeYearsStartDate",
        "http://dbpedia.org/ontology/birthDate",
        "http://dbpedia.org/ontology/capitalCoordinates",
        "http://dbpedia.org/ontology/circle",
        "http://dbpedia.org/ontology/creationYear",
        "http://dbpedia.org/ontology/date",
        "http://dbpedia.org/ontology/deathDate",
        "http://dbpedia.org/ontology/deathYear",
        "http://dbpedia.org/ontology/firstAscentYear",
        "http://dbpedia.org/ontology/foundingYear",
        "http://dbpedia.org/ontology/openingDate",
        "http://dbpedia.org/ontology/range",
        "http://dbpedia.org/ontology/rating",
        "http://dbpedia.org/ontology/releaseDate",
        "http://dbpedia.org/ontology/timeZone",
        "http://dbpedia.org/ontology/water",
        "http://dbpedia.org/ontology/year",

        # Msr
        "http://dbpedia.org/ontology/areaTotal",
        "http://dbpedia.org/ontology/assets",
        "http://dbpedia.org/ontology/bedCount",
        "http://dbpedia.org/ontology/broadcastArea",
        "http://dbpedia.org/ontology/catholicPercentage",
        "http://dbpedia.org/ontology/collectionSize",
        "http://dbpedia.org/ontology/cost",
        "http://dbpedia.org/ontology/currency",
        "http://dbpedia.org/ontology/depth",
        "http://dbpedia.org/ontology/duration",
        "http://dbpedia.org/ontology/elevation",
        "http://dbpedia.org/ontology/floorCount",
        "http://dbpedia.org/ontology/frequency",
        "http://dbpedia.org/ontology/GeopoliticalOrganisation/populationDensity",
        "http://dbpedia.org/ontology/giniCoefficient",
        "http://dbpedia.org/ontology/grossDomesticProduct",
        "http://dbpedia.org/ontology/income",
        "http://dbpedia.org/ontology/infantMortality",
        "http://dbpedia.org/ontology/Lake/volume",
        "http://dbpedia.org/ontology/length",
        "http://dbpedia.org/ontology/lifeExpectancy",
        "http://dbpedia.org/ontology/number",
        "http://dbpedia.org/ontology/numberOfEmployees",
        "http://dbpedia.org/ontology/numberOfPlayers",
        "http://dbpedia.org/ontology/numberOfVisitors",
        "http://dbpedia.org/ontology/perCapitaIncome",
        "http://dbpedia.org/ontology/Person/height",
        "http://dbpedia.org/ontology/Person/weight",
        "http://dbpedia.org/ontology/PopulatedPlace/area",
        "http://dbpedia.org/ontology/PopulatedPlace/areaTotal",
        "http://dbpedia.org/ontology/PopulatedPlace/populationDensity",
        "http://dbpedia.org/ontology/populationTotal",
        "http://dbpedia.org/ontology/revenue",
        "http://dbpedia.org/ontology/sales",
        "http://dbpedia.org/ontology/Software/fileSize",
        "http://dbpedia.org/ontology/statisticValue",

    ],
    "T2Dv1": [
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://dbpedia.org/ontology/language",
        "http://dbpedia.org/ontology/industry",
        "http://dbpedia.org/ontology/country",
        "http://dbpedia.org/ontology/currency",
        "http://dbpedia.org/ontology/populationTotal",
        "http://dbpedia.org/ontology/location",
        "http://dbpedia.org/ontology/currencyCode",
        "http://dbpedia.org/ontology/releaseDate",
        "http://dbpedia.org/ontology/city",
        "http://dbpedia.org/ontology/capital",
        "http://dbpedia.org/ontology/commonName",
        "http://dbpedia.org/ontology/revenue",
        "http://dbpedia.org/ontology/director",
        "http://dbpedia.org/ontology/elevation",
        "http://dbpedia.org/ontology/address",
        "http://dbpedia.org/ontology/collectionSize",
        "http://dbpedia.org/ontology/Work/runtime",
        "http://dbpedia.org/ontology/sales",
        "http://dbpedia.org/ontology/publisher",
        "http://dbpedia.org/ontology/assets",
        "http://dbpedia.org/ontology/PopulatedPlace/area",
        "http://dbpedia.org/ontology/genre",
        "http://dbpedia.org/ontology/cost",
        "http://dbpedia.org/ontology/GeopoliticalOrganisation/populationDensity",
        "http://dbpedia.org/ontology/federalState",
        "http://dbpedia.org/ontology/priceMoney",
        "http://dbpedia.org/ontology/computingPlatform",
        "http://dbpedia.org/ontology/symbol",
        "http://dbpedia.org/ontology/iataLocationIdentifier",
        "http://dbpedia.org/ontology/numberOfEmployees",
        "http://dbpedia.org/ontology/artist",
        "http://dbpedia.org/ontology/populationMetro",
        "http://dbpedia.org/ontology/grossDomesticProduct",
        "http://dbpedia.org/ontology/developer",
        "http://dbpedia.org/ontology/company",
        "http://dbpedia.org/ontology/category",
        "http://dbpedia.org/ontology/result",
        "http://dbpedia.org/ontology/PopulatedPlace/populationDensity",
        "http://dbpedia.org/ontology/owner",
        "http://dbpedia.org/ontology/mayor",
        "http://dbpedia.org/ontology/age",
        "http://dbpedia.org/ontology/iso31661Code",
        "http://dbpedia.org/ontology/family",
        "http://dbpedia.org/ontology/year",
        "http://dbpedia.org/ontology/netIncome",
        "http://dbpedia.org/ontology/governmentType",
        "http://dbpedia.org/ontology/description",
        "http://dbpedia.org/ontology/conflict",
        "http://dbpedia.org/ontology/circle",
        "http://dbpedia.org/ontology/zipCode",
        "http://dbpedia.org/ontology/type",
        "http://dbpedia.org/ontology/Software/fileSize",
        "http://dbpedia.org/ontology/rating",
        "http://dbpedia.org/ontology/Person/height",
        "http://dbpedia.org/ontology/mountainRange",
        "http://dbpedia.org/ontology/formerName",
        "http://dbpedia.org/ontology/date",
        "http://dbpedia.org/ontology/author",
        "http://dbpedia.org/ontology/starring",
        "http://dbpedia.org/ontology/sexualOrientation",
        "http://dbpedia.org/ontology/marketCapitalisation",
        "http://dbpedia.org/ontology/kinOfLanguage",
        "http://dbpedia.org/ontology/hairColor",
        "http://dbpedia.org/ontology/genus",
        "http://dbpedia.org/ontology/foundingYear",
        "http://dbpedia.org/ontology/child",
        "http://dbpedia.org/ontology/bodyStyle",
        "http://dbpedia.org/ontology/areaTotal",
        "http://dbpedia.org/ontology/officialLanguage",
        "http://dbpedia.org/ontology/isoCode",
        "http://dbpedia.org/ontology/grades",
        "http://dbpedia.org/ontology/gender",
        "http://dbpedia.org/ontology/eyeColor",
        "http://dbpedia.org/ontology/command",
        "http://dbpedia.org/ontology/code",
        "http://dbpedia.org/ontology/club",
        "http://dbpedia.org/ontology/class",
        "http://dbpedia.org/ontology/birthYear",
        "http://dbpedia.org/ontology/birthDate",
        "http://dbpedia.org/ontology/alternativeTitle",
        "http://dbpedia.org/ontology/album",
        "http://dbpedia.org/ontology/teamName",
        "http://dbpedia.org/ontology/synonym",
        "http://dbpedia.org/ontology/subdivision",
        "http://dbpedia.org/ontology/rank",
        "http://dbpedia.org/ontology/producer",
        "http://dbpedia.org/ontology/populationAsOf",
        "http://dbpedia.org/ontology/Person/weight",
        "http://dbpedia.org/ontology/otherName",
        "http://dbpedia.org/ontology/origin",
        "http://dbpedia.org/ontology/numberOfSpeakers",
        "http://dbpedia.org/ontology/manufacturer",
        "http://dbpedia.org/ontology/length",
        "http://dbpedia.org/ontology/ingredient",
        "http://dbpedia.org/ontology/height",
        "http://dbpedia.org/ontology/growingGrape",
        "http://dbpedia.org/ontology/floorCount",
        "http://dbpedia.org/ontology/eventDate",
        "http://dbpedia.org/ontology/ethnicity",
        "http://dbpedia.org/ontology/duration",
        "http://dbpedia.org/ontology/creationYear",
        "http://dbpedia.org/ontology/continent",
        "http://dbpedia.org/ontology/alias",
        "http://dbpedia.org/ontology/topLevelDomain",
        "http://dbpedia.org/ontology/team",
        "http://dbpedia.org/ontology/soccerTournamentTopScorer",
        "http://dbpedia.org/ontology/runtime",
        "http://dbpedia.org/ontology/religion",
        "http://dbpedia.org/ontology/reference",
        "http://dbpedia.org/ontology/recordLabel",
        "http://dbpedia.org/ontology/ranking",
        "http://dbpedia.org/ontology/range",
        "http://dbpedia.org/ontology/programmeFormat",
        "http://dbpedia.org/ontology/phonePrefix",
        "http://dbpedia.org/ontology/person",
        "http://dbpedia.org/ontology/organisation",
        "http://dbpedia.org/ontology/operatingSystem",
        "http://dbpedia.org/ontology/openingDate",
        "http://dbpedia.org/ontology/numberOfVisitors",
        "http://dbpedia.org/ontology/numberOfPages",
        "http://dbpedia.org/ontology/number",
        "http://dbpedia.org/ontology/nationality",
        "http://dbpedia.org/ontology/narrator",
        "http://dbpedia.org/ontology/name",
        "http://dbpedia.org/ontology/musicBy",
        "http://dbpedia.org/ontology/membership",
        "http://dbpedia.org/ontology/locatedInArea",
        "http://dbpedia.org/ontology/lifeExpectancy",
        "http://dbpedia.org/ontology/license",
        "http://dbpedia.org/ontology/latestReleaseVersion",
        "http://dbpedia.org/ontology/isbn",
        "http://dbpedia.org/ontology/internationalPhonePrefix",
        "http://dbpedia.org/ontology/infantMortality",
        "http://dbpedia.org/ontology/income",
        "http://dbpedia.org/ontology/imageSize",
        "http://dbpedia.org/ontology/iataAirlineCode",
        "http://dbpedia.org/ontology/headquarter",
        "http://dbpedia.org/ontology/frequency",
        "http://dbpedia.org/ontology/firstPublicationDate",
        "http://dbpedia.org/ontology/firstAscentYear",
        "http://dbpedia.org/ontology/fipsCode",
        "http://dbpedia.org/ontology/fileSize",
        "http://dbpedia.org/ontology/education",
        "http://dbpedia.org/ontology/distributingCompany",
        "http://dbpedia.org/ontology/day",
        "http://dbpedia.org/ontology/creator",
        "http://dbpedia.org/ontology/county",
        "http://dbpedia.org/ontology/coordinates",
        "http://dbpedia.org/ontology/conservationStatus",
        "http://dbpedia.org/ontology/colorChart",
        "http://dbpedia.org/ontology/certification",
        "http://dbpedia.org/ontology/catholicPercentage",
        "http://dbpedia.org/ontology/capitalCoordinates",
        "http://dbpedia.org/ontology/capacity",
        "http://dbpedia.org/ontology/brand",
        "http://dbpedia.org/ontology/birthPlace",
        "http://dbpedia.org/ontology/bedCount",
        "http://dbpedia.org/ontology/analogChannel",
        "http://dbpedia.org/ontology/activeYearsStartDate",
        "http://dbpedia.org/ontology/abbreviation",
        "http://dbpedia.org/ontology/writer",
        "http://dbpedia.org/ontology/wikiPageExternalLink",
        "http://dbpedia.org/ontology/width",
        "http://dbpedia.org/ontology/weight",
        "http://dbpedia.org/ontology/water",
        "http://dbpedia.org/ontology/waistSize",
        "http://dbpedia.org/ontology/voltageOfElectrification",
        "http://dbpedia.org/ontology/usk",
        "http://dbpedia.org/ontology/translatedMotto",
        "http://dbpedia.org/ontology/tradeMark",
        "http://dbpedia.org/ontology/timeZone",
        "http://dbpedia.org/ontology/technique",
        "http://dbpedia.org/ontology/status",
        "http://dbpedia.org/ontology/statisticValue",
        "http://dbpedia.org/ontology/state",
        "http://dbpedia.org/ontology/spouse",
        "http://dbpedia.org/ontology/species",
        "http://dbpedia.org/ontology/sex",
        "http://dbpedia.org/ontology/sessionNumber",
        "http://dbpedia.org/ontology/service",
        "http://dbpedia.org/ontology/seatingCapacity",
        "http://dbpedia.org/ontology/requirement",
        "http://dbpedia.org/ontology/relative",
        "http://dbpedia.org/ontology/registration",
        "http://dbpedia.org/ontology/region",
        "http://dbpedia.org/ontology/recordDate",
        "http://dbpedia.org/ontology/ratio",
        "http://dbpedia.org/ontology/railwayRollingStock",
        "http://dbpedia.org/ontology/projectCoordinator",
        "http://dbpedia.org/ontology/programmingLanguage",
        "http://dbpedia.org/ontology/profession",
        "http://dbpedia.org/ontology/product",
        "http://dbpedia.org/ontology/position",
        "http://dbpedia.org/ontology/portrayer",
        "http://dbpedia.org/ontology/plays",
        "http://dbpedia.org/ontology/Planet/averageSpeed",
        "http://dbpedia.org/ontology/piercing",
        "http://dbpedia.org/ontology/pictureFormat",
        "http://dbpedia.org/ontology/personName",
        "http://dbpedia.org/ontology/personFunction",
        "http://dbpedia.org/ontology/percentageAlcohol",
        "http://dbpedia.org/ontology/perCapitaIncome",
        "http://dbpedia.org/ontology/part",
        "http://dbpedia.org/ontology/originalName",
        "http://dbpedia.org/ontology/originalLanguage",
        "http://dbpedia.org/ontology/order",
        "http://dbpedia.org/ontology/operator",
        "http://dbpedia.org/ontology/numberOfTracks",
        "http://dbpedia.org/ontology/numberOfPlayers",
        "http://dbpedia.org/ontology/numberOfMembers",
        "http://dbpedia.org/ontology/numberOfHouses",
        "http://dbpedia.org/ontology/notes",
        "http://dbpedia.org/ontology/nickname",
        "http://dbpedia.org/ontology/nameDay",
        "http://dbpedia.org/ontology/musicalArtist",
        "http://dbpedia.org/ontology/movement",
        "http://dbpedia.org/ontology/model",
        "http://dbpedia.org/ontology/locationIdentifier",
        "http://dbpedia.org/ontology/locationCity",
        "http://dbpedia.org/ontology/literaryGenre",
        "http://dbpedia.org/ontology/licenceNumber",
        "http://dbpedia.org/ontology/latestReleaseDate",
        "http://dbpedia.org/ontology/latestPreviewVersion",
        "http://dbpedia.org/ontology/languageCode",
        "http://dbpedia.org/ontology/Lake/volume",
        "http://dbpedia.org/ontology/knownFor",
        "http://dbpedia.org/ontology/jockey",
        "http://dbpedia.org/ontology/id",
        "http://dbpedia.org/ontology/icaoLocationIdentifier",
        "http://dbpedia.org/ontology/hipSize",
        "http://dbpedia.org/ontology/highestState",
        "http://dbpedia.org/ontology/highestRank",
        "http://dbpedia.org/ontology/heritageRegister",
        "http://dbpedia.org/ontology/giniCoefficient",
        "http://dbpedia.org/ontology/generalManager",
        "http://dbpedia.org/ontology/frequencyOfPublication",
        "http://dbpedia.org/ontology/frenchName",
        "http://dbpedia.org/ontology/founder",
        "http://dbpedia.org/ontology/format",
        "http://dbpedia.org/ontology/flyingHours",
        "http://dbpedia.org/ontology/firstPublicationYear",
        "http://dbpedia.org/ontology/fastestLap",
        "http://dbpedia.org/ontology/executiveProducer",
        "http://dbpedia.org/ontology/englishName",
        "http://dbpedia.org/ontology/editor",
        "http://dbpedia.org/ontology/doctoralAdvisor",
        "http://dbpedia.org/ontology/diocese",
        "http://dbpedia.org/ontology/depth",
        "http://dbpedia.org/ontology/deliveryDate",
        "http://dbpedia.org/ontology/deathYear",
        "http://dbpedia.org/ontology/deathDate",
        "http://dbpedia.org/ontology/dateLastUpdated",
        "http://dbpedia.org/ontology/dateAgreement",
        "http://dbpedia.org/ontology/cuisine",
        "http://dbpedia.org/ontology/credit",
        "http://dbpedia.org/ontology/constructionMaterial",
        "http://dbpedia.org/ontology/construction",
        "http://dbpedia.org/ontology/component",
        "http://dbpedia.org/ontology/collection",
        "http://dbpedia.org/ontology/classis",
        "http://dbpedia.org/ontology/classification",
        "http://dbpedia.org/ontology/careerPrizeMoney",
        "http://dbpedia.org/ontology/careerPoints",
        "http://dbpedia.org/ontology/campusType",
        "http://dbpedia.org/ontology/bustSize",
        "http://dbpedia.org/ontology/Building/floorArea",
        "http://dbpedia.org/ontology/broadcastNetwork",
        "http://dbpedia.org/ontology/broadcastArea",
        "http://dbpedia.org/ontology/blazon",
        "http://dbpedia.org/ontology/bestFinish",
        "http://dbpedia.org/ontology/AutomobileEngine/width",
        "http://dbpedia.org/ontology/AutomobileEngine/weight",
        "http://dbpedia.org/ontology/AutomobileEngine/length",
        "http://dbpedia.org/ontology/AutomobileEngine/height",
        "http://dbpedia.org/ontology/AutomobileEngine/diameter",
        "http://dbpedia.org/ontology/area",
        "http://dbpedia.org/ontology/architecturalStyle",
        "http://dbpedia.org/ontology/architect",
        "http://dbpedia.org/ontology/almaMater",
        "http://dbpedia.org/ontology/affiliation",
        "http://dbpedia.org/ontology/affiliate",
        "http://dbpedia.org/ontology/activity",
        "http://dbpedia.org/ontology/activeYearsEndDate",
        "http://dbpedia.org/ontology/activeYear",
        "http://dbpedia.org/ontology/acquirementDate",

    ],
    "Vendor":
        []
}
MsrTypeMapping = {
    "vendor":
        {"Money-Price": SubCategory.Money,
         "Money-Income": SubCategory.Money,
         "Money-Outcome": SubCategory.Money,
         "Money-Others": SubCategory.Money,
         "Length-Distance": SubCategory.Spatial_Length,
         "Length-Height": SubCategory.Spatial_Length,
         "Length-Width": SubCategory.Spatial_Length,
         "Length-Depth": SubCategory.Spatial_Length,
         "Temperature": SubCategory.Thermodynamics_Temperature,
         "Area": SubCategory.Spatial_Area,
         "Weight": SubCategory.Matter_Mass,
         "Volume": SubCategory.Spatial_Volume,  # TODO: Volume sound or saptial
         "Occurrences": SubCategory.Unitless_Quantity,
         "Quantity": SubCategory.Unitless_Quantity,
         "Elapsed time": SubCategory.Time_TimeInterval,
         "Frequency": SubCategory.Time_Frequency,
         "Speed": SubCategory.Mechanics_Speed,
         "Acceleration": SubCategory.Others,  # TODO: name is not sure
         "Population": SubCategory.Unitless_Quantity,
         "Power": SubCategory.Ele_Power,
         "Age": SubCategory.Time_TimeInterval,
         "Growth Rate": SubCategory.Unitless_Ratio,
         "Network flow": SubCategory.Data,
         "Flow rate": SubCategory.Unitless_Ratio,
         "Other": SubCategory.Others,
         },
    "ReVendor":
        {
            "Length": SubCategory.Spatial_Length,
            "Length1": SubCategory.Spatial_Length,
            "Width": SubCategory.Spatial_Length,
            "Elevation": SubCategory.Spatial_Length,
            "Depth": SubCategory.Spatial_Length,
            "height": SubCategory.Spatial_Length,
            "Area": SubCategory.Spatial_Area,
            "Volume/capacity": SubCategory.Spatial_Volume,
            "Volume": SubCategory.Spatial_Volume,
            "Time interval (duration)": SubCategory.Time_TimeInterval,
            "Duration": SubCategory.Time_TimeInterval,
            "Day": SubCategory.Time_TimeInterval,
            "Week": SubCategory.Time_TimeInterval,
            "Month": SubCategory.Time_TimeInterval,
            "Year": SubCategory.Time_TimeInterval,
            "Age": SubCategory.Time_TimeInterval,
            "Runtime": SubCategory.Time_TimeInterval,
            "Frequency": SubCategory.Time_Frequency,
            "Mass(Weight)": SubCategory.Matter_Mass,
            "Mass": SubCategory.Matter_Mass,
            "Density": SubCategory.Others,
            "Concentration": SubCategory.Others,
            "Temperature": SubCategory.Thermodynamics_Temperature,
            "Pressure": SubCategory.Thermodynamics_Pressure,
            "Thermal capacity": SubCategory.Others,
            "Thermal conductivity": SubCategory.Others,
            "Currency": SubCategory.Money,
            "Sale": SubCategory.Money,
            "Asset": SubCategory.Money,
            "Income": SubCategory.Money,
            "Revenue": SubCategory.Money,
            "Cost": SubCategory.Money,
            "GDP": SubCategory.Money,
            "Financial": SubCategory.Money,
            "Data/file size": SubCategory.Data,
            "Count/number": SubCategory.Unitless_Quantity,
            "population": SubCategory.Unitless_Quantity,
            "Rating": SubCategory.Unitless_Score,
            "Rank": SubCategory.Unitless_Rank,
            "Score": SubCategory.Unitless_Score,
            "Ratio": SubCategory.Unitless_Ratio,
            "Percentage": SubCategory.Unitless_Ratio,
            "Rate": SubCategory.Unitless_Ratio,
            "Others": SubCategory.Others,
            "Bool": -1,
            "Speed": SubCategory.Mechanics_Speed,
            "Kinematic viscosity": SubCategory.Others,
            "Energy": SubCategory.Ele_Energy,
            "Power": SubCategory.Ele_Power,
            "voltageOfElectrification": SubCategory.Others,
            "Electrical conductivity": SubCategory.Others,
            "Luminosity": SubCategory.Others,
            "Enthalpy": SubCategory.Others,
            "dim": -1,
            'Others/angle': SubCategory.Unitless_Angle,
            "Dimension": -1,
            "Chemistry": SubCategory.Others,
            "Count": SubCategory.Unitless_Quantity,
            'Unclear': -1,
            'Money': SubCategory.Money,
            'Unclear (Unknown)': -1,
            'Indicator': SubCategory.Unitless_Score,
            "Factor": SubCategory.Unitless_Factor,
            "Unknown": -1,
            "Indicator/Score": SubCategory.Unitless_Score,
            "Weight": SubCategory.Matter_Mass,
            "Population": SubCategory.Unitless_Quantity,
            "Electric current": SubCategory.Others,
            "Angle": SubCategory.Unitless_Angle,
            "Sci-Chemistry": SubCategory.Others,
            "Volume / Capacity": SubCategory.Spatial_Volume,
            "Count (Amount)": SubCategory.Unitless_Quantity,
            "Dim": -1,
            "Indicator (Index)": SubCategory.Unitless_Score,
            'Factor / Coefficient': SubCategory.Unitless_Factor,
            'Longtitude/Lattitude': SubCategory.Unitless_Angle,
            'Grade': SubCategory.Unitless_Score,

        },
    "T2Dv1":
        {
            "http://dbpedia.org/ontology/currency": SubCategory.Money,
            "http://dbpedia.org/ontology/populationTotal": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/revenue": SubCategory.Money,
            "http://dbpedia.org/ontology/elevation": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/collectionSize": SubCategory.Unitless_Quantity,  # TODO: check
            "http://dbpedia.org/ontology/Work/runtime": SubCategory.Time_TimeInterval,
            "http://dbpedia.org/ontology/sales": SubCategory.Money,
            "http://dbpedia.org/ontology/assets": SubCategory.Money,
            "http://dbpedia.org/ontology/PopulatedPlace/area": SubCategory.Spatial_Area,
            "http://dbpedia.org/ontology/cost": SubCategory.Money,
            "http://dbpedia.org/ontology/GeopoliticalOrganisation/populationDensity": SubCategory.Others,
            "http://dbpedia.org/ontology/priceMoney": SubCategory.Money,
            "http://dbpedia.org/ontology/numberOfEmployees": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/grossDomesticProduct": SubCategory.Money,
            "http://dbpedia.org/ontology/result": SubCategory.Unitless_Quantity,  # TODO:check
            "http://dbpedia.org/ontology/PopulatedPlace/populationDensity": SubCategory.Others,
            "http://dbpedia.org/ontology/age": SubCategory.Time_TimeInterval,
            "http://dbpedia.org/ontology/year": SubCategory.Time_TimeInterval,
            "http://dbpedia.org/ontology/netIncome": SubCategory.Money,
            "http://dbpedia.org/ontology/Software/fileSize": SubCategory.Data,
            "http://dbpedia.org/ontology/rating": SubCategory.Unitless_Score,
            "http://dbpedia.org/ontology/Person/height": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/mountainRange": SubCategory.Others,  # TODO: check
            "http://dbpedia.org/ontology/marketCapitalisation": SubCategory.Money,
            "http://dbpedia.org/ontology/foundingYear": SubCategory.Time_TimeInterval,  # TODO:dimension?
            "http://dbpedia.org/ontology/areaTotal": SubCategory.Spatial_Area,
            "http://dbpedia.org/ontology/rank": SubCategory.Unitless_Rank,
            "http://dbpedia.org/ontology/populationAsOf": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/Person/weight": SubCategory.Matter_Mass,
            "http://dbpedia.org/ontology/numberOfSpeakers": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/length": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/height": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/floorCount": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/duration": SubCategory.Time_TimeInterval,
            "http://dbpedia.org/ontology/soccerTournamentTopScorer": SubCategory.Unitless_Score,
            "http://dbpedia.org/ontology/runtime": SubCategory.Time_TimeInterval,
            "http://dbpedia.org/ontology/ranking": SubCategory.Unitless_Rank,
            "http://dbpedia.org/ontology/range": SubCategory.Others,
            "http://dbpedia.org/ontology/numberOfVisitors": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/numberOfPages": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/number": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/lifeExpectancy": SubCategory.Time_TimeInterval,
            "http://dbpedia.org/ontology/infantMortality": SubCategory.Unitless_Ratio,
            "http://dbpedia.org/ontology/income": SubCategory.Money,
            "http://dbpedia.org/ontology/imageSize": SubCategory.Data,  # TODO:check
            "http://dbpedia.org/ontology/frequency": SubCategory.Time_Frequency,
            "http://dbpedia.org/ontology/fileSize": SubCategory.Data,
            "http://dbpedia.org/ontology/day": SubCategory.Time_TimeInterval,
            "http://dbpedia.org/ontology/catholicPercentage": SubCategory.Unitless_Ratio,
            "http://dbpedia.org/ontology/capacity": SubCategory.Spatial_Volume,
            "http://dbpedia.org/ontology/bedCount": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/width": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/weight": SubCategory.Matter_Mass,
            "http://dbpedia.org/ontology/waistSize": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/voltageOfElectrification": SubCategory.Others,
            "http://dbpedia.org/ontology/statisticValue": SubCategory.Others,
            "http://dbpedia.org/ontology/seatingCapacity": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/ratio": SubCategory.Unitless_Ratio,
            "http://dbpedia.org/ontology/plays": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/Planet/averageSpeed": SubCategory.Mechanics_Speed,
            "http://dbpedia.org/ontology/percentageAlcohol": SubCategory.Unitless_Ratio,
            "http://dbpedia.org/ontology/perCapitaIncome": SubCategory.Money,
            "http://dbpedia.org/ontology/numberOfTracks": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/numberOfPlayers": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/numberOfMembers": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/numberOfHouses": SubCategory.Unitless_Quantity,
            "http://dbpedia.org/ontology/Lake/volume": SubCategory.Spatial_Volume,
            "http://dbpedia.org/ontology/hipSize": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/highestRank": SubCategory.Unitless_Rank,
            "http://dbpedia.org/ontology/giniCoefficient": SubCategory.Others,  # TODO:Financial others?
            "http://dbpedia.org/ontology/frequencyOfPublication": SubCategory.Time_Frequency,
            "http://dbpedia.org/ontology/flyingHours": SubCategory.Time_TimeInterval,
            "http://dbpedia.org/ontology/depth": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/credit": SubCategory.Money,  # TODO:xuefen or daikuan
            "http://dbpedia.org/ontology/careerPrizeMoney": SubCategory.Money,
            "http://dbpedia.org/ontology/careerPoints": SubCategory.Unitless_Score,
            "http://dbpedia.org/ontology/bustSize": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/Building/floorArea": SubCategory.Spatial_Area,
            "http://dbpedia.org/ontology/AutomobileEngine/width": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/AutomobileEngine/weight": SubCategory.Matter_Mass,
            "http://dbpedia.org/ontology/AutomobileEngine/length": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/AutomobileEngine/height": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/AutomobileEngine/diameter": SubCategory.Spatial_Length,
            "http://dbpedia.org/ontology/area": SubCategory.Spatial_Area,

        },
    "semtab":
        {
            "Height": SubCategory.Spatial_Length,
            "Width": SubCategory.Spatial_Length,
            "Angle": SubCategory.Unitless_Angle,
            "Length1": SubCategory.Spatial_Length,
            "Others (Known)": SubCategory.Others,
            "Count (Amount)": SubCategory.Unitless_Quantity,
            "Sales": SubCategory.Money,
            "Pressure": SubCategory.Thermodynamics_Pressure,
            "Speed": SubCategory.Mechanics_Speed,
            "Elevation": SubCategory.Spatial_Length,
            "Population": SubCategory.Unitless_Quantity,
            "Area": SubCategory.Spatial_Area,
            "Others": SubCategory.Others,
            "Density": SubCategory.Others,
            "Longtitude/Lattitude": SubCategory.Unitless_Angle,
            "Duration": SubCategory.Time_TimeInterval,
            "Ratio": SubCategory.Unitless_Ratio,
            "Mass (Weight)": SubCategory.Matter_Mass,
            "Depth": SubCategory.Spatial_Length,
            "Rating": SubCategory.Unitless_Score,
            "Voltage": SubCategory.Others,
            "GDP": SubCategory.Money,
            "Age": SubCategory.Time_TimeInterval,
            "Income": SubCategory.Money,
            "Money": SubCategory.Money,
            "Indicator (Index)": SubCategory.Unitless_Score,
            "Temperature": SubCategory.Thermodynamics_Temperature,
            "Percentage": SubCategory.Unitless_Ratio,
            "Luminosity": SubCategory.Others,
            "Length": SubCategory.Spatial_Length,
            "Frequency": SubCategory.Time_Frequency,
            "Energy": SubCategory.Ele_Energy,
            "Concentration": SubCategory.Others,
            "Cost": SubCategory.Money,
            "Unclear (Unknown)": -1,
            "Rank": SubCategory.Unitless_Rank,
            "Enthalpy": SubCategory.Others,
            "Volume/Capacity": SubCategory.Spatial_Volume,
            "Data/file size": SubCategory.Data,
            "Asset": SubCategory.Money,
            "Score": SubCategory.Unitless_Score,
            "ChangeRate": SubCategory.Unitless_Ratio,
            "Performance Score": SubCategory.Unitless_Score,
            "Power": SubCategory.Ele_Power,
            "Electric": SubCategory.Others,
            "Count": SubCategory.Unitless_Quantity
        }
}
