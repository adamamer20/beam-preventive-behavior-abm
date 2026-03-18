"""Survey prediction data model for LLM behavioral analysis."""

from typing import Annotated

from pydantic import BaseModel, Field


class SurveyPrediction(BaseModel):
    """Pydantic model for structured LLM survey prediction output.

    This model defines the schema for survey responses predicted by LLMs,
    including trust in institutions, vaccine opinions, and protective behaviors.
    """

    # TODO: covax_reason should be handled in a special way
    """covax_reason: Annotated[
        str, Field(description="Why did or didn't you get vaccinated for COVID-19?")
    ]"""

    institut_trust_vax_1: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Trust in information about vaccines from the local government",
        ),
    ]
    institut_trust_vax_2: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Trust in information about vaccines from the national government",
        ),
    ]
    institut_trust_vax_3: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Trust in information about vaccines from the family doctor",
        ),
    ]
    institut_trust_vax_4: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Trust in information about vaccines from the local health authorities",
        ),
    ]
    institut_trust_vax_5: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Trust in information about vaccines from the national health authorities",
        ),
    ]
    institut_trust_vax_6: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Trust in information about vaccines from the World Health Organization",
        ),
    ]
    institut_trust_vax_7: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Trust in information about vaccines from religious leaders",
        ),
    ]
    institut_trust_vax_8: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Trust in information about vaccines from influencers and other public personalities",
        ),
    ]
    opinion_1: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Vaccines are decisive for the protection of human health",
        ),
    ]
    fiveC_1: Annotated[
        int,
        Field(ge=1, le=7, description="I am completely confident that vaccines are safe"),
    ]
    fiveC_2: Annotated[
        int,
        Field(
            ge=1,
            le=7,
            description="Vaccination is unnecessary because vaccine preventable diseases are not common anymore",
        ),
    ]
    fiveC_3: Annotated[
        int,
        Field(
            ge=1,
            le=7,
            description="Everyday stress prevents me from getting vaccinated",
        ),
    ]
    fiveC_4: Annotated[
        int,
        Field(
            ge=1,
            le=7,
            description="When I think about getting vaccinated, I weigh benefits and risks to make the best decision possible",
        ),
    ]
    fiveC_5: Annotated[
        int,
        Field(
            ge=1,
            le=7,
            description="When everyone is vaccinated, I don't need to get vaccinated",
        ),
    ]
    protective_behaviour_1: Annotated[
        int,
        Field(
            ge=1,
            le=7,
            description="Safe and effective vaccines should be made mandatory for the entire population",
        ),
    ]
    protective_behaviour_2: Annotated[
        int,
        Field(
            ge=1,
            le=7,
            description="I use masks in crowded places when experiencing respiratory symptoms",
        ),
    ]
    protective_behaviour_3: Annotated[
        int,
        Field(
            ge=1,
            le=7,
            description="I stay at home when experiencing respiratory symptoms",
        ),
    ]
    protective_behaviour_4: Annotated[
        int,
        Field(
            ge=1,
            le=7,
            description="When the epidemic pressure is high, I use masks in crowded places",
        ),
    ]
