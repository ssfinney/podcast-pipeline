# Issue Spec: Milestone Events & Notable Moments Extractor

**Status:** Backlog / Scheduled for Next Milestone  
**Target:** Podcast & Church Service Historical Chronicle  
**Tracking Issue:** `issues/01-milestone-events-extractor.md`

---

## 1. Overview & Objective

Every Sunday service recording contains not only theological teaching, but also real-time historical milestones, church family moments, community happenings, and notable reports:
- **Ministry Spotlights:** Kairos Prison Ministry updates, youth conventions, mission trip reports.
- **Church Family Milestones:** Baby dedications, baptisms, marriages, member testimonies (*I Am Christ Chapel* series).
- **Special Guests & Keynotes:** Visiting ministers, guest speakers (e.g., Rabbi Greg, Bonnie Wilson, Terry Theis).
- **Notable News & Announcements:** Building/facility campaigns, semester discipleship launches, conference dates, major community events.
- **Answered Prayers & Miracle Reports:** Physical healings, salvation numbers, life turnaround testimonies shared from stage.

The objective of this module is to automatically extract, categorize, and compile a searchable chronological timeline (`MILESTONE_TIMELINE.md` / `milestones.json`) across all 720 episodes.

---

## 2. Event Taxonomy & Classification

Each detected milestone event will be classified under one of the following standard categories:

| Category | Description | Example Triggers / Phrases |
|---|---|---|
| `MISSION_OUTREACH` | Prison ministries, homeless outreach, global missions, local charity. | *"Kairos weekend", "mission trip to", "food drive", "recidivism rate"* |
| `FAMILY_LIFECYCLE` | Baby dedications, child dedications, baptisms, weddings, funerals. | *"Baby dedication", "stand with the family", "baptized today", "dedicated to the Lord"* |
| `SPECIAL_GUEST` | Guest preachers, visiting evangelists, member testimonies. | *"Welcome Bonnie Wilson", "Rabbi Greg next Sunday", "Katie Peck sharing today"* |
| `TESTIMONY_REPORT` | Miracles, personal conversions, spiritual transformations. | *"12 gave their life to the Lord", "he literally died and came back", "cancer-free"* |
| `COMMUNITY_NEWS` | Conferences, youth retreats, discipleship classes, life groups. | *"Family discipleship kicking off", "Go and Go conference", "girls retreat"* |
| `LEADERSHIP_VISION` | Building updates, annual vision, pastoral announcements, budget reports. | *"State of the church", "building campaign", "expanding our sanctuary"* |

---

## 3. Extraction Architecture & Data Schema

### Extraction Pipeline:
1. **Target Slice:** The module focuses on the **Preshow / Announcements / Giving / Welcome** segment (`00:00:00` $\to$ `preaching_start`) plus the **Dismissal / Post-Sermon** segment (`preaching_end` $\to$ end of recording), as well as key personal stories cited in sermons.
2. **Gemini Schema-Driven Parsing:**
   ```json
   {
     "date": "YYYY-MM-DD",
     "episode_title": "...",
     "milestones": [
       {
         "category": "MISSION_OUTREACH | FAMILY_LIFECYCLE | SPECIAL_GUEST | TESTIMONY_REPORT | COMMUNITY_NEWS | LEADERSHIP_VISION",
         "event_title": "Kairos Women's Prison Ministry September Launch",
         "summary": "Bonnie Wilson shared ministry statistics (recidivism drop from 23% to 10%) and called for volunteers for the Sept 10-13 weekend.",
         "key_people": ["Bonnie Wilson", "Sandra", "John C. Wood"],
         "timestamp_start": "00:30:00",
         "timestamp_end": "00:50:00",
         "key_quotes": ["We are entering a place of darkness and bringing in the light of Christ."],
         "date_of_upcoming_event": "2026-09-10"
       }
     ]
   }
   ```

---

## 4. Planned Outputs

1. **`MILESTONE_TIMELINE.md`**:
   - Chronological table and narrative history of Christ Chapel Macon over the years.
   - Allows searching for *"When was Baby Maggie dedicated?"* or *"What dates did Kairos minister in 2026?"*
2. **`milestones.json`**:
   - Structured JSON database linked by `guid` and timestamp to the exact audio recordings.
3. **NotebookLM Direct Source**:
   - `MILESTONE_TIMELINE.md` can be loaded into NotebookLM as a dedicated church history source for deep Q&A.

---

## 5. Next Steps for Implementation

- [ ] Create `milestones.py` with structured schema extraction.
- [ ] Add `--milestones` flag to `pipeline.py`.
- [ ] Backfill milestone extraction across all transcribed episodes.
